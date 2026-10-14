"""Redressement de l'inclinaison résiduelle (quelques degrés) par profil de projection.

Quand les lignes de texte et les traits de tableau sont parfaitement horizontaux, le profil des lignes
(somme de l'encre par ligne de pixels) alterne brutalement entre pics et creux. On fait tourner le masque
d'encre sur une plage d'angles et on garde l'angle qui maximise la somme des carrés des variations du
profil : recherche grossière (pas de 0,25°) puis fine (pas de 0,02°) autour du meilleur angle.
"""
import cv2
import numpy as np

from .encre import masque_encre

PLAGE = 5.0            # inclinaison maximale cherchée (degrés, de part et d'autre de 0)
PAS_GROSSIER = 0.25
PAS_FIN = 0.02


def _tourner_masque(encre, angle):
    """Tourne un masque d'angle degrés (sens antihoraire), sans agrandir le cadre."""
    h, w = encre.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(encre, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def score_horizontalite(encre, angle):
    """Somme des carrés des variations du profil des lignes après rotation : maximale quand le texte est à plat."""
    profil = _tourner_masque(encre, angle).sum(axis=1).astype(np.float64)
    return float(np.sum(np.diff(profil) ** 2))


def _meilleur_angle(encre, angles):
    """Angle de la liste qui maximise le score d'horizontalité."""
    scores = [score_horizontalite(encre, a) for a in angles]
    return float(angles[int(np.argmax(scores))]), scores


def estimer_inclinaison(gris, plage=PLAGE):
    """Renvoie (angle, details) : angle (degrés, sens antihoraire) à appliquer pour mettre le texte à plat."""
    encre = masque_encre(gris)
    grossier, scores = _meilleur_angle(encre, np.arange(-plage, plage + 1e-9, PAS_GROSSIER))
    fin, _ = _meilleur_angle(encre, np.arange(grossier - PAS_GROSSIER, grossier + PAS_GROSSIER + 1e-9, PAS_FIN))
    contraste = max(scores) / (np.median(scores) + 1e-9)
    return round(fin, 2), dict(contraste_profil=round(float(contraste), 2))


def redresser(image, angle, fond=255):
    """Tourne l'image d'angle degrés (sens antihoraire) en agrandissant le cadre pour ne rien couper ;
    les zones ajoutées sont remplies de blanc."""
    if abs(angle) < 1e-6:
        return image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nl, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    m[0, 2] += nl / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(image, m, (nl, nh), flags=cv2.INTER_CUBIC, borderValue=fond)
