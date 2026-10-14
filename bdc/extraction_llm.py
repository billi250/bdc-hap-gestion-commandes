"""Extraction par modèle à vision : lire_scan() envoie les pages prétraitées au modèle de config.yaml, avec
le schéma imposé ; extraire_champs() ajoute les corrections déterministes (référentiel, formats).

Le modèle lit la page entière, imprimé et manuscrit, et rend pour chaque valeur sa confiance et sa boîte sur
la page. Les recalages sur le référentiel (nom du client, codes d'analyse) sont faits ici, en Python, et
tracés dans « recalages » : on sait toujours ce que le modèle a lu et ce qui a été corrigé.
"""
import re
from difflib import SequenceMatcher

from .champs import sans_accents
from .modeles import png
from .schema import BonDeCommande

CONSIGNE = """Tu lis un bon de commande d'analyses de sols envoyé au laboratoire. Il peut avoir plusieurs pages,
fournies dans l'ordre. Recopie exactement ce qui est écrit, imprimé ou manuscrit, sans rien inventer ni
compléter : si une information est absente ou illisible, valeur = null et confiance = 0.

Rôles :
- emetteur_* : l'organisme dont l'en-tête figure en haut du bon, qui passe la commande (nom, adresse, SIRET,
  téléphone, e-mail). Ce n'est jamais le laboratoire destinataire.
- charge_affaires : la personne indiquée « chargé d'affaires » (ou technicien, responsable du dossier).
- client_* : le client final ou le site pour qui sont faites les analyses (« client final », « raison
  sociale », « exploitant », « site »), avec adresse du site, parcelles, commune, téléphone, interlocuteur.
  S'il n'y a pas de client final distinct, laisse ces champs à null.
- laboratoire : le laboratoire destinataire du bon.
- nb_echantillons_annonce : seulement si un nombre d'échantillons est écrit sur le bon ; ne le compte pas.

Formats : dates JJ/MM/AAAA (convertis une date écrite en lettres) ; téléphones 0X XX XX XX XX ; SIRET en
14 chiffres sans espace ; délai = la case cochée (standard, normal, urgent ou express).
Attention aux chiffres manuscrits : relis chaque chiffre un par un (1 et 7, 3 et 8, 2 et 9, 0 et 6).

Tableau des échantillons : une entrée par ligne remplie, dans l'ordre. analyses = codes des analyses demandées,
lus sur les cases cochées ou le texte, parmi : {codes}. ligne_barree = true si la ligne est rayée d'un trait ;
recopie quand même son contenu. Une ligne vide n'est pas un échantillon.

Confiance (0 à 100) : sois honnête ; 95 et plus seulement si la lecture est sans aucun doute, moins de 70
si un caractère est incertain. Boîte : [ymin, xmin, ymax, xmax] de la valeur (pas de l'étiquette), entiers de
0 à 1000 relatifs à la page indiquée dans « page »."""


def consigne(referentiel):
    """Consigne complète, avec la liste des codes d'analyse du référentiel."""
    return CONSIGNE.replace("{codes}", ", ".join(a["code"] for a in referentiel["analyses"]))


FORMES_JURIDIQUES = r"\b(sas|sasu|sarl|sa|eurl|earl|scea|gaec|sci|scop|eirl)\b"


def proche(valeur, connues, seuil=0.82):
    """Valeur connue la plus proche (sans accents, ponctuation ni forme juridique ajoutée) : nom connu suivi
    d'un complément (secteur, ville), sinon ressemblance au-delà du seuil."""
    def cle(t):
        t = re.sub(r"[^a-z0-9]+", " ", sans_accents(t)).strip()
        sans_forme = re.sub(FORMES_JURIDIQUES, " ", t).split()
        return " ".join(sans_forme) or t
    if not valeur:
        return None
    debuts = [c for c in connues if cle(valeur).startswith(cle(c) + " ")]   # « ECOTELLUS ENVIRONNEMENT »
    if debuts:
        return max(debuts, key=lambda c: len(cle(c)))
    meilleur = max(connues, key=lambda c: SequenceMatcher(None, cle(valeur), cle(c)).ratio(), default=None)
    if meilleur and SequenceMatcher(None, cle(valeur), cle(meilleur)).ratio() >= seuil:
        return meilleur
    return None


def code_analyse(texte, referentiel):
    """Code d'analyse du référentiel correspondant à un texte lu (« HAP16 », « matière sèche »), ou None."""
    cle = re.sub(r"[^a-z0-9]", "", sans_accents(texte or ""))
    for a in referentiel["analyses"]:
        formes = [a["code"], a["code_lims"]] + list(a.get("synonymes", []))
        if cle in {re.sub(r"[^a-z0-9]", "", sans_accents(f)) for f in formes}:
            return a["code"]
    return None


def recaler(bdc, referentiel):
    """Corrections déterministes après lecture. Modifie bdc et renvoie la liste des recalages tracés."""
    recalages = []

    def noter(champ, avant, apres, raison):
        if avant != apres:
            recalages.append(dict(champ=champ, lu=avant, retenu=apres, raison=raison))

    clients = [c["nom"] for c in referentiel.get("clients", [])]
    for nom, connues in (("emetteur_nom", clients), ("laboratoire", referentiel.get("laboratoires", []))):
        champ = getattr(bdc, nom)
        connu = proche(champ.valeur, connues)
        if connu:
            noter(nom, champ.valeur, connu, "nom du référentiel le plus proche")
            champ.valeur = connu
    siret = bdc.emetteur_siret
    if siret.valeur:
        chiffres = re.sub(r"\D", "", siret.valeur)
        noter("emetteur_siret", siret.valeur, chiffres, "espaces retirés")
        siret.valeur = chiffres
    for i, e in enumerate(bdc.echantillons, 1):
        codes = []
        for a in e.analyses:
            code = code_analyse(a, referentiel)
            noter(f"echantillons[{i}].analyses", a, code or a, "code du référentiel")
            codes.append(code or a)
        e.analyses = list(dict.fromkeys(codes))
    return recalages


def pages_png(pages):
    """Pages prétraitées (niveaux de gris) encodées en PNG pour l'envoi au modèle."""
    return [png(p) for p in pages]


def lire_scan(pages, passerelle, referentiel, document="", justification="extraction d'un BDC reçu"):
    """Lecture brute par le modèle : (BonDeCommande validé par le schéma, infos sur le modèle utilisé)."""
    return passerelle.generer_json("extraction", consigne(referentiel), pages_png(pages), BonDeCommande,
                                   document=document, justification=justification)


def extraire_champs(pages, passerelle, referentiel, document=""):
    """Lecture par le modèle puis recalages. Renvoie (BonDeCommande, infos) ; infos contient les recalages."""
    bdc, infos = lire_scan(pages, passerelle, referentiel, document)
    infos["lu_par_le_modele"] = bdc.model_dump()
    infos["recalages"] = recaler(bdc, referentiel)
    return bdc, infos
