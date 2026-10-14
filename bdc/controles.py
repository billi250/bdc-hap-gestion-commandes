"""Contrôles automatiques, score de confiance, doublons et alertes délai.

Chaque anomalie est un dict(champ, niveau, message) : niveau « erreur » (à corriger avant validation) ou
« attention » (à regarder). Les champs en anomalie sont affichés en orange dans l'écran de revue, et l'agent
de relecture commence par eux.
"""
import re
from datetime import date, datetime, timedelta

from .champs import luhn
from .extraction_llm import code_analyse, proche
from .reconstruction import annule_par_observation
from .schema import CHAMPS, DELAIS, OBLIGATOIRES

TELEPHONE = re.compile(r"^0[1-9](\d{2}){4}$")
EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


def anomalie(champ, niveau, message):
    return dict(champ=champ, niveau=niveau, message=message)


def lire_date(texte):
    """Date JJ/MM/AAAA (ou JJ/MM/AA) en objet date ; None si le texte n'est pas une date réelle."""
    m = re.fullmatch(r"\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{2}|\d{4})\s*", texte or "")
    if not m:
        return None
    j, mo, a = (int(x) for x in m.groups())
    try:
        return date(a + 2000 if a < 100 else a, mo, j)
    except ValueError:
        return None


def annule(e):
    """Vrai si l'échantillon est annulé : ligne barrée, ou « annulé » dans les observations."""
    return bool(e.get("ligne_barree")) or annule_par_observation(e.get("observations"))


def _valeur(bdc, nom):
    return ((bdc.get(nom) or {}).get("valeur") or "").strip()


def controle_formats(bdc, seuils, aujourd_hui):
    """Champs obligatoires, dates, SIRET, téléphones, e-mail, délai."""
    a = [anomalie(n, "erreur", f"champ obligatoire manquant : {CHAMPS[n][1]}") for n in OBLIGATOIRES
         if not _valeur(bdc, n)]
    for nom in ("date_commande", "date_prelevement"):
        texte = _valeur(bdc, nom)
        if not texte:
            continue
        d = lire_date(texte)
        if d is None:
            a.append(anomalie(nom, "erreur", f"date impossible : « {texte} »"))
        elif nom == "date_commande" and d > aujourd_hui + timedelta(days=seuils["date_max_jours_futur"]):
            a.append(anomalie(nom, "erreur", f"date de commande dans le futur : {texte}"))
        elif nom == "date_commande" and d < aujourd_hui - timedelta(days=seuils["date_max_jours_passe"]):
            a.append(anomalie(nom, "attention", f"date de commande de plus d'un an : {texte}"))
    dc, dp = lire_date(_valeur(bdc, "date_commande")), lire_date(_valeur(bdc, "date_prelevement"))
    if dc and dp and dp > dc + timedelta(days=90):
        a.append(anomalie("date_prelevement", "attention", "prélèvement plus de 90 jours après la commande"))
    siret = re.sub(r"\D", "", _valeur(bdc, "emetteur_siret"))
    if siret and (len(siret) != 14 or not luhn(siret)):
        a.append(anomalie("emetteur_siret", "attention",
                          f"SIRET {siret} : {'pas 14 chiffres' if len(siret) != 14 else 'clé de Luhn invalide'}"))
    for nom in ("emetteur_telephone", "client_telephone"):
        tel = re.sub(r"[\s.-]", "", _valeur(bdc, nom))
        if tel and not TELEPHONE.match(tel):
            a.append(anomalie(nom, "attention", f"téléphone mal formé : {_valeur(bdc, nom)}"))
    mail = _valeur(bdc, "emetteur_email")
    if mail and not EMAIL.match(mail.replace(" ", "")):
        a.append(anomalie("emetteur_email", "attention", f"e-mail mal formé : {mail}"))
    delai = _valeur(bdc, "delai").lower()
    if delai and delai not in DELAIS:
        a.append(anomalie("delai", "attention", f"délai non reconnu : {delai}"))
    return a


def controle_referentiel(bdc, referentiel):
    """Client et laboratoire connus, codes d'analyse connus."""
    a = []
    clients = [c["nom"] for c in referentiel.get("clients", [])]
    emetteur = _valeur(bdc, "emetteur_nom")
    if emetteur and emetteur not in clients:
        suggestion = proche(emetteur, clients, seuil=0.6)
        a.append(anomalie("emetteur_nom", "attention", f"client inconnu du référentiel : {emetteur}"
                          + (f" (proche de {suggestion})" if suggestion else " (nouveau client ?)")))
    labo = _valeur(bdc, "laboratoire")
    if labo and labo not in referentiel.get("laboratoires", []):
        a.append(anomalie("laboratoire", "attention", f"laboratoire inattendu : {labo}"))
    for i, e in enumerate(bdc.get("echantillons") or [], 1):
        for code in e.get("analyses") or []:
            if code_analyse(code, referentiel) is None:
                a.append(anomalie(f"echantillons[{i}]", "erreur", f"échantillon {i} : analyse inconnue « {code} »"))
    return a


def controle_echantillons(bdc):
    """Tableau : présent, références uniques, analyses demandées, nombre annoncé = nombre listé."""
    ech = bdc.get("echantillons") or []
    actifs = [e for e in ech if not annule(e)]
    if not ech:
        return [anomalie("echantillons", "erreur", "aucun échantillon lu")]
    a = []
    refs = [(e.get("reference") or "").strip() for e in ech]
    for i, (e, ref) in enumerate(zip(ech, refs), 1):
        if not ref:
            a.append(anomalie(f"echantillons[{i}]", "erreur", f"échantillon {i} sans référence"))
        elif refs.count(ref) > 1:
            a.append(anomalie(f"echantillons[{i}]", "erreur", f"référence en double : {ref}"))
        if not annule(e) and not e.get("analyses"):
            a.append(anomalie(f"echantillons[{i}]", "erreur", f"échantillon {i} ({ref}) : aucune analyse demandée"))
    annonce = re.sub(r"\D", "", _valeur(bdc, "nb_echantillons_annonce"))
    if annonce and int(annonce) not in (len(actifs), len(ech)):
        a.append(anomalie("nb_echantillons_annonce", "erreur",
                          f"{annonce} échantillons annoncés, {len(actifs)} listés ({len(ech) - len(actifs)} annulés)"))
    return a


def controle_confiance(bdc, seuil):
    """Lectures peu sûres selon le modèle (confiance sous le seuil)."""
    a = []
    for nom in CHAMPS:
        c = bdc.get(nom) or {}
        if c.get("valeur") and c.get("confiance", 100) < seuil:
            a.append(anomalie(nom, "attention", f"lecture peu sûre ({c['confiance']}) : {CHAMPS[nom][1]}"))
    for i, e in enumerate(bdc.get("echantillons") or [], 1):
        if e.get("confiance", 100) < seuil:
            a.append(anomalie(f"echantillons[{i}]", "attention", f"échantillon {i} : lecture peu sûre ({e['confiance']})"))
    return a


def controle_doublons(bdc, sha256, autres):
    """autres : BDC déjà en base (dict id, fichier, sha256, numero, emetteur). Même fichier, ou même numéro
    de commande chez le même émetteur (fax puis courrier) : doublon."""
    a = []
    numero, emetteur = _valeur(bdc, "numero_commande"), _valeur(bdc, "emetteur_nom")
    for o in autres:
        if sha256 and o.get("sha256") == sha256:
            a.append(anomalie("doublon", "erreur", f"fichier identique au BDC n° {o['id']} ({o['fichier']})"))
        elif numero and o.get("numero") == numero and (o.get("emetteur") or "") == emetteur:
            a.append(anomalie("doublon", "erreur", f"même n° de commande {numero} que le BDC n° {o['id']} "
                                                  f"({o['fichier']}) : reçu deux fois ?"))
    return a


def controler(bdc, referentiel, seuils, sha256=None, autres=(), aujourd_hui=None):
    """Tous les contrôles d'un BDC (dict issu de BonDeCommande.model_dump()). Renvoie la liste des anomalies."""
    aujourd_hui = aujourd_hui or date.today()
    return (controle_formats(bdc, seuils, aujourd_hui) + controle_referentiel(bdc, referentiel)
            + controle_echantillons(bdc) + controle_confiance(bdc, seuils["confiance_min"])
            + controle_doublons(bdc, sha256, autres))


def score(bdc, anomalies):
    """Score de confiance du BDC sur 100 : confiance moyenne des valeurs lues, moins 10 par erreur et 3 par
    point d'attention. Sert à trier la file « à vérifier » (les plus douteux d'abord)."""
    confiances = [(bdc.get(n) or {}).get("confiance", 0) for n in CHAMPS if (bdc.get(n) or {}).get("valeur")]
    confiances += [e.get("confiance", 0) for e in bdc.get("echantillons") or []]
    base = sum(confiances) / len(confiances) if confiances else 0
    erreurs = sum(a["niveau"] == "erreur" for a in anomalies)
    return round(max(0.0, min(100.0, base - 10 * erreurs - 3 * (len(anomalies) - erreurs))), 1)


def statut_initial(anomalies):
    """« a verifier » s'il y a au moins une anomalie, sinon « brouillon » (prêt à valider)."""
    return "a verifier" if anomalies else "brouillon"


def alertes_delai(bdc_liste, seuils, maintenant=None):
    """BDC non validés urgents ou en attente depuis trop longtemps. bdc_liste : dict avec recu_le (ISO),
    statut, delai, numero, emetteur, id. Renvoie une liste de dict(id, numero, emetteur, attente_jours, raison)."""
    maintenant = maintenant or datetime.now().astimezone()
    sortie = []
    for b in bdc_liste:
        if b["statut"] == "valide":
            continue
        attente = (maintenant - datetime.fromisoformat(b["recu_le"])).total_seconds() / 86400
        urgent = (b.get("delai") or "") in ("urgent", "express")
        if urgent and attente >= seuils["jours_urgent"]:
            raison = f"{b['delai']} et non validé depuis {attente:.1f} j"
        elif attente >= seuils["jours_attente_alerte"]:
            raison = f"en attente depuis {attente:.1f} j"
        elif urgent:
            raison = f"{b['delai']} : à traiter en priorité"
        else:
            continue
        sortie.append(dict(id=b["id"], numero=b.get("numero"), emetteur=b.get("emetteur"),
                           attente_jours=round(attente, 1), raison=raison))
    return sorted(sortie, key=lambda s: -s["attente_jours"])
