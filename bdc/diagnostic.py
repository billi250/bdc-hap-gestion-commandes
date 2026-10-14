"""Diagnostic du défaut principal d'une page, et réglages de normalisation adaptés.

Chaque défaut laisse une signature simple dans l'image d'origine (seuils calés sur les 28 documents du corpus) :
- photo : bordure nettement plus sombre que le centre (feuille posée sur un fond) ;
- fax : image quasi binaire (presque tous les pixels noirs ou blancs), noir pur, nombreux points isolés ;
- papier teinté (liasse carbone) : le fond du papier n'est pas blanc ;
- photocopie : aucun pixel vraiment noir et beaucoup de petites taches (grain de copie) ;
- encre pâle : aucun pixel vraiment noir (impression claire, crayon) ;
- tache ou ombre locale : le fond estimé présente un creux étendu ;
- propre : aucune de ces signatures.
"""
import cv2
import numpy as np

from .normalisation import estimer_fond
from .perspective import est_photo

SEUILS = dict(fax_extremes=0.96, fax_noir=10, fax_points=0.5, fond_teinte=245, photocopie_noir=130,
              photocopie_points=0.35, pale_noir=95, creux=0.04, creux_aire=0.005)

# Réglages qui diffèrent des valeurs par défaut, par défaut diagnostiqué (les autres restent inchangés).
REGLAGES = {
    "fax": dict(debruitage_h=12),
    "encre pale": dict(constante_binarisation=10),
    "photocopie": dict(constante_binarisation=12),
    "papier teinte": dict(constante_binarisation=12),
}


def indices(gris):
    """Indicateurs mesurés sur l'image d'origine en niveaux de gris."""
    _, encre = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(encre, connectivity=8)
    fond = estimer_fond(gris).astype(np.float32)
    lisse = cv2.GaussianBlur(estimer_fond(gris, 25, 61).astype(np.float32), (0, 0), 15)
    h, w = lisse.shape
    centre = lisse[int(.05 * h):int(.95 * h), int(.05 * w):int(.95 * w)]
    creux = (np.median(centre) - centre) / max(float(np.median(centre)), 1.0)
    return dict(photo=bool(est_photo(gris)), noir_p1=float(np.percentile(gris, 1)),
                extremes=float(((gris < 40) | (gris > 215)).mean()), fond_median=float(np.median(fond)),
                points=float((stats[1:, cv2.CC_STAT_AREA] <= 6).sum() / max(n - 1, 1)),
                aire_creux=float((creux > SEUILS["creux"]).mean()))


def classer(ind):
    """Défaut principal à partir des indicateurs (règles appliquées dans cet ordre)."""
    s = SEUILS
    if ind["photo"]:
        return "photo"
    if ind["extremes"] >= s["fax_extremes"] and ind["noir_p1"] <= s["fax_noir"] and ind["points"] >= s["fax_points"]:
        return "fax"
    if ind["fond_median"] < s["fond_teinte"]:
        return "papier teinte"
    if ind["noir_p1"] >= s["photocopie_noir"] and ind["points"] >= s["photocopie_points"]:
        return "photocopie"
    if ind["noir_p1"] >= s["pale_noir"]:
        return "encre pale"
    if ind["aire_creux"] >= s["creux_aire"]:
        return "tache ou ombre"
    return "propre"


def diagnostiquer(gris):
    """Renvoie (défaut, réglages adaptés, indicateurs arrondis)."""
    ind = indices(gris)
    defaut = classer(ind)
    return defaut, dict(REGLAGES.get(defaut, {})), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ind.items()}
