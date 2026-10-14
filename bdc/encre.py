"""Masque d'encre réduit, commun à l'orientation et au redressement."""
import cv2
import numpy as np

from .normalisation import retirer_fond

COTE_MAX = 1200  # les estimations d'angle se font sur une image réduite (rapide, assez précis)


def reduire(gris, cote_max=COTE_MAX):
    """Réduit l'image pour que son plus grand côté fasse au plus cote_max pixels. Renvoie (image, facteur)."""
    f = min(1.0, cote_max / max(gris.shape))
    if f == 1.0:
        return gris, 1.0
    return cv2.resize(gris, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), f


def masque_encre(gris, cote_max=COTE_MAX):
    """Masque binaire (1 = encre) de l'image réduite, après retrait du fond et seuil d'Otsu."""
    petit, _ = reduire(gris, cote_max)
    propre = retirer_fond(petit, dilatation=9, mediane=31)
    _, encre = cv2.threshold(propre, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return encre


def retirer_traits(encre, fraction=25):
    """Retire les longs traits horizontaux et verticaux (cadres, tableaux) pour ne garder que le texte."""
    h, w = encre.shape
    horiz = cv2.morphologyEx(encre, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // fraction, 3), 1)))
    vert = cv2.morphologyEx(encre, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // fraction, 3))))
    return np.clip(encre.astype(np.int16) - horiz - vert, 0, 1).astype(np.uint8)
