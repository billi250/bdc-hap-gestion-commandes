"""Fusion des deux lectures d'un bon : modèle vision-langage (meilleur sur le manuscrit) et Tesseract (très sûr sur
l'imprimé). Le modèle vision-langage ne donne pas de score de confiance : l'accord des deux moteurs en tient lieu.

Règles :
- champs numériques (téléphone, SIRET, numéro, dates, parcelles) : accord seulement si les chiffres sont
  identiques, car une erreur d'un seul chiffre est justement celle qui compte ;
- valeur retenue : celle de Tesseract s'il l'a lue avec une confiance >= 90 (texte imprimé), sinon celle du
  modèle vision-langage ;
- confiance : 95 si les deux lectures sont identiques, 80 si elles sont proches, 55 si elles diffèrent (le champ
  est alors signalé, avec les deux lectures).
"""
import re
from difflib import SequenceMatcher

from .champs import sans_accents

NUMERIQUES = {"client_telephone", "emetteur_telephone", "emetteur_siret", "numero_commande", "date_commande",
              "date_prelevement", "parcelles"}
SEUIL_PROCHE, CONFIANCE_TESSERACT_SURE = 0.85, 90.0
CONFIANCE_IDENTIQUE, CONFIANCE_PROCHE, CONFIANCE_DIFFERENTE, CONFIANCE_SEUL = 95.0, 80.0, 55.0, 70.0


def _normal(valeur):
    """Forme comparable : minuscules sans accents, chiffres et lettres seulement."""
    return re.sub(r"[^a-z0-9]", "", sans_accents(str(valeur)))


def accord(a, b, numerique=False):
    """Similarité (0 à 1) de deux lectures ; pour un champ numérique, 1 si les chiffres sont identiques, sinon 0."""
    if numerique:
        ca, cb = re.sub(r"\D", "", str(a)), re.sub(r"\D", "", str(b))
        return 1.0 if ca and ca == cb else 0.0
    a, b = _normal(a), _normal(b)
    if a and b and (a in b or b in a):          # « ECOTELLUS » et « ECOTELLUS ENVIRONNEMENT » : même valeur, abrégée
        return max(SEUIL_PROCHE, SequenceMatcher(None, a, b).ratio())
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def fusionner_champ(nom, v, t):
    """Fusion d'un champ lu par les deux moteurs. Renvoie (champ, alerte ou None)."""
    sim = accord(v["valeur"], t["valeur"], nom in NUMERIQUES)
    identique = _normal(v["valeur"]) == _normal(t["valeur"]) or (nom in NUMERIQUES and sim == 1.0)
    garde_tesseract = (t.get("confiance") or 0) >= CONFIANCE_TESSERACT_SURE
    retenu, autre = (t, v) if garde_tesseract else (v, t)
    confiance = CONFIANCE_IDENTIQUE if identique else CONFIANCE_PROCHE if sim >= SEUIL_PROCHE else CONFIANCE_DIFFERENTE
    champ = dict(retenu, confiance=confiance, autre_lecture=autre["valeur"], accord=round(sim, 2),
                 source="tesseract" if garde_tesseract else "modele vision-langage")
    alerte = None
    if confiance == CONFIANCE_DIFFERENTE:
        alerte = f"{nom} : lectures différentes (modèle « {v['valeur']} », Tesseract « {t['valeur']} »), à vérifier"
    return champ, alerte


def fusionner_champs(vlm, tesseract):
    """Renvoie (champs fusionnés, alertes). Chaque champ porte la valeur retenue, l'autre lecture et l'accord."""
    champs, alertes = {}, []
    for nom in sorted(set(vlm) | set(tesseract)):
        v, t = vlm.get(nom), tesseract.get(nom)
        if isinstance(v, list) or isinstance(t, list):
            champs[nom] = v or t
        elif v and t:
            champs[nom], alerte = fusionner_champ(nom, v, t)
            if alerte:
                alertes.append(alerte)
        elif v or t:
            seul = v or t
            champs[nom] = dict(seul, confiance=seul.get("confiance") or CONFIANCE_SEUL, autre_lecture=None, accord=None,
                               source="tesseract" if t else "modele vision-langage")
        else:
            champs[nom] = None
    return champs, alertes
