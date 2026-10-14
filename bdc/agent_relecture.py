"""Agent de relecture : extraire, contrôler, relire les champs douteux, recontrôler.

Boucle (au plus `tours` passages) :
1. les contrôles listent les anomalies (règle échouée, confiance basse) ;
2. l'agent choisit les champs à relire (erreurs d'abord, puis confiance la plus basse), au plus
   `relectures_max` par document ;
3. pour chacun, il découpe la zone de la valeur sur la page (boîte renvoyée par le modèle, avec une marge),
   l'agrandit et demande au modèle de relire ce seul champ, en lui donnant la règle qui a échoué ;
4. la nouvelle valeur n'est retenue que si sa confiance est meilleure et qu'elle ne crée pas d'anomalie ;
5. on recontrôle ; on s'arrête quand plus rien ne s'améliore.

Chaque étape est tracée (plan, action, résultat) pour être affichée au laborantin : rien n'est caché.
"""
import cv2

from .controles import controler
from .modeles import png
from .schema import CHAMPS, Relecture

MARGE = 0.04           # marge autour de la boîte, en fraction de la page
ZOOM_MIN_PX = 900      # largeur minimale du découpage envoyé

CONSIGNE_RELECTURE = """Voici un agrandissement d'une zone d'un bon de commande d'analyses de sols.
Relis uniquement le champ « {libelle} » ({groupe}). Lecture précédente : « {avant} ».
Problème signalé : {probleme}.
Recopie exactement ce qui est écrit, chiffre par chiffre, sans corriger ni deviner. Format attendu : {format}.
Si la zone ne contient pas ce champ ou s'il est illisible, valeur = null."""

FORMATS = {"date_commande": "JJ/MM/AAAA", "date_prelevement": "JJ/MM/AAAA", "emetteur_siret": "14 chiffres",
           "emetteur_telephone": "0X XX XX XX XX", "client_telephone": "0X XX XX XX XX",
           "delai": "standard, normal, urgent ou express", "nb_echantillons_annonce": "un nombre"}


def decouper(page, boite, marge=MARGE):
    """Zone [ymin, xmin, ymax, xmax] (0-1000) de la page, avec marge, agrandie pour la lecture."""
    h, w = page.shape[:2]
    y0, x0, y1, x1 = boite
    y0, y1 = max(0, int((y0 / 1000 - marge) * h)), min(h, int((y1 / 1000 + marge) * h))
    x0, x1 = max(0, int((x0 / 1000 - 2 * marge) * w)), min(w, int((x1 / 1000 + 2 * marge) * w))
    zone = page[y0:y1, x0:x1]
    if zone.size == 0:
        return None
    f = max(1.0, ZOOM_MIN_PX / zone.shape[1])
    return cv2.resize(zone, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if f > 1 else zone


def choisir(bdc, anomalies, deja, limite):
    """Champs d'en-tête à relire : en anomalie, avec une boîte, pas encore relus. Erreurs d'abord."""
    candidats = {}
    for a in anomalies:
        nom = a["champ"]
        champ = bdc.get(nom) if nom in CHAMPS else None
        if not champ or not champ.get("boite") or nom in deja:
            continue
        rang = (0 if a["niveau"] == "erreur" else 1, champ.get("confiance", 0))
        if nom not in candidats or rang < candidats[nom][0]:
            candidats[nom] = (rang, a["message"])
    return [(n, m) for n, (_, m) in sorted(candidats.items(), key=lambda kv: kv[1][0])][:limite]


def relire_champ(nom, probleme, bdc, pages, passerelle, document=""):
    """Relit un champ sur un zoom de sa zone. Renvoie (Relecture, image envoyée) ou (None, None)."""
    champ = bdc[nom]
    page = pages[min(len(pages), max(1, champ.get("page", 1))) - 1]
    zone = decouper(page, champ["boite"])
    if zone is None:
        return None, None
    groupe, libelle = CHAMPS[nom]
    texte = CONSIGNE_RELECTURE.format(libelle=libelle, groupe=groupe, avant=champ.get("valeur") or "vide",
                                      probleme=probleme, format=FORMATS.get(nom, "texte tel qu'écrit"))
    r, _ = passerelle.generer_json("relecture", texte, [png(zone)], Relecture, document=document,
                                   justification=f"relecture ciblée du champ {nom} (zone découpée)")
    return r, zone


def nb_anomalies(anomalies, nom):
    return sum(a["champ"] == nom for a in anomalies)


def relire(bdc, pages, passerelle, referentiel, seuils, document="", controle=None, tours=2, rappel=None):
    """Boucle de l'agent. bdc : dict (BonDeCommande.model_dump()), modifié sur place. pages : pages
    prétraitées envoyées à l'extraction. controle(bdc) -> anomalies (par défaut : controles.controler).
    rappel(étape) : appelé à chaque étape (affichage en direct). Renvoie (anomalies finales, journal)."""
    controle = controle or (lambda b: controler(b, referentiel, seuils))
    journal, relus = [], set()
    anomalies = controle(bdc)

    def noter(etape):
        journal.append(etape)
        if rappel:
            rappel(etape)

    for tour in range(1, tours + 1):
        plan = choisir(bdc, anomalies, relus, seuils["relectures_max"] - len(relus))
        noter(dict(type="plan", tour=tour, anomalies=len(anomalies), champs=[n for n, _ in plan],
                   message=(f"Tour {tour} : {len(anomalies)} anomalie(s), relecture de "
                            f"{', '.join(CHAMPS[n][1] for n, _ in plan)}") if plan else
                   f"Tour {tour} : {len(anomalies)} anomalie(s), aucun champ relisable : arrêt"))
        if not plan:
            break
        ameliore = False
        for nom, probleme in plan:
            relus.add(nom)
            avant = dict(bdc[nom])
            try:
                r, _ = relire_champ(nom, probleme, bdc, pages, passerelle, document)
            except Exception as e:                       # un champ non relu n'arrête pas l'agent
                noter(dict(type="action", champ=nom, avant=avant.get("valeur"), apres=None, retenu=False,
                           message=f"{CHAMPS[nom][1]} : relecture impossible ({e})"))
                continue
            if r is None or not r.valeur:
                noter(dict(type="action", champ=nom, avant=avant.get("valeur"), apres=None, retenu=False,
                           message=f"{CHAMPS[nom][1]} : rien de lisible dans la zone"))
                continue
            essai = {**bdc, nom: {**avant, "valeur": r.valeur, "confiance": r.confiance}}
            nouvelles = controle(essai)
            mieux = (r.valeur != avant.get("valeur") and r.confiance >= avant.get("confiance", 0)
                     and nb_anomalies(nouvelles, nom) <= nb_anomalies(anomalies, nom))
            confirme = r.valeur == avant.get("valeur")
            if mieux:
                bdc[nom] = essai[nom]
                anomalies, ameliore = nouvelles, True
            elif confirme:
                bdc[nom] = {**avant, "confiance": max(avant.get("confiance", 0), r.confiance)}
                anomalies = controle(bdc)
            noter(dict(type="action", champ=nom, avant=avant.get("valeur"), apres=r.valeur, retenu=mieux,
                       confiance=r.confiance, modele=passerelle.reglage("relecture")["modele"],
                       message=(f"{CHAMPS[nom][1]} : « {avant.get('valeur')} » → « {r.valeur} » ({r.confiance}) retenu"
                                if mieux else f"{CHAMPS[nom][1]} : lecture confirmée ({r.confiance})" if confirme
                                else f"{CHAMPS[nom][1]} : relu « {r.valeur} » ({r.confiance}), non retenu")
                       + (f". {r.explication}" if r.explication else "")))
        if not ameliore:
            noter(dict(type="fin", message="Plus rien ne s'améliore : fin de la relecture"))
            break
    noter(dict(type="bilan", anomalies=len(anomalies),
               message=f"Bilan : {len(anomalies)} anomalie(s) restante(s) pour le laborantin"))
    return anomalies, journal
