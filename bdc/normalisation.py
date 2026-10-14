"""Normalisation de l'image : débruitage, atténuation du fond (ombres, taches), binarisation locale."""
import cv2
import numpy as np

PARAMS_DEFAUT = dict(debruitage_h=10, fond_dilatation=15, fond_mediane=41, bloc_binarisation=31,
                     constante_binarisation=15, seuil_aplat=90, aire_point_max=6, rayon_isolement=6,
                     marge_bord=0.03)


def debruiter(gris, h=PARAMS_DEFAUT["debruitage_h"]):
    """Débruitage par moyennes non locales (préserve mieux les traits fins qu'un flou)."""
    return cv2.fastNlMeansDenoising(gris, None, h=h, templateWindowSize=7, searchWindowSize=21)


def estimer_fond(gris, dilatation=PARAMS_DEFAUT["fond_dilatation"], mediane=PARAMS_DEFAUT["fond_mediane"]):
    """Estime le fond clair de la feuille : la dilatation efface l'encre (sombre et fine), la médiane
    lisse le résultat. Les ombres et les taches larges restent dans le fond estimé."""
    noyau = cv2.getStructuringElement(cv2.MORPH_RECT, (dilatation, dilatation))
    return cv2.medianBlur(cv2.dilate(gris, noyau), mediane)


def retirer_fond(gris, **params):
    """Divise l'image par son fond estimé : un fond irrégulier (ombre, café, papier carbone) devient
    uniformément blanc, l'encre garde son contraste local."""
    fond = estimer_fond(gris, **params).astype(np.float32)
    return cv2.divide(gris.astype(np.float32), np.maximum(fond, 1), scale=255).clip(0, 255).astype(np.uint8)


def binariser(gris, bloc=PARAMS_DEFAUT["bloc_binarisation"], constante=PARAMS_DEFAUT["constante_binarisation"]):
    """Binarisation adaptative (seuil gaussien local) : encre noire (0) sur fond blanc (255)."""
    return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bloc, constante)


def remplir_aplats(binaire, gris, seuil=PARAMS_DEFAUT["seuil_aplat"]):
    """Le seuil local « évide » les grands aplats noirs (bandeaux de titre) : un pixel franchement sombre
    dans l'image normalisée est remis à l'encre, quel que soit le seuil local."""
    sortie = binaire.copy()
    sortie[gris < seuil] = 0
    return sortie


def _deux_points(centres, dx_max=3, dy=(4, 16)):
    """Pour chaque petit point, vrai s'il a un autre petit point presque à la verticale (écart horizontal
    <= dx_max, vertical dans dy) : c'est la forme d'un deux-points ou d'un point-virgule."""
    resultat = np.zeros(len(centres), dtype=bool)
    for debut in range(0, len(centres), 1000):   # par paquets, pour borner la mémoire
        c = centres[debut:debut + 1000]
        ecart_x = np.abs(c[:, None, 0] - centres[None, :, 0])
        ecart_y = np.abs(c[:, None, 1] - centres[None, :, 1])
        resultat[debut:debut + 1000] = ((ecart_x <= dx_max) & (ecart_y >= dy[0]) & (ecart_y <= dy[1])).any(axis=1)
    return resultat


def retirer_points(binaire, aire_max=PARAMS_DEFAUT["aire_point_max"], rayon=PARAMS_DEFAUT["rayon_isolement"]):
    """Supprime le bruit de fax : une petite composante (aire <= aire_max) est retirée si aucune composante
    plus grande (du texte) ne se trouve à moins de rayon pixels, sauf si elle forme avec un autre petit point
    un deux-points (points alignés verticalement). Les points de bruit et ceux des deux-points ont la même
    taille à 150 dpi : seul le contexte permet de les distinguer."""
    n, etiquettes, stats, centres = cv2.connectedComponentsWithStats(255 - binaire, connectivity=8)
    petit = stats[:, cv2.CC_STAT_AREA] <= aire_max
    petit[0] = False
    texte = ((etiquettes > 0) & ~petit[etiquettes]).astype(np.uint8)
    voisinage = cv2.dilate(texte, np.ones((2 * rayon + 1, 2 * rayon + 1), np.uint8)).astype(bool)
    proche = np.bincount(etiquettes[voisinage], minlength=n) > 0
    a_retirer = np.where(petit & ~proche)[0]
    a_retirer = a_retirer[~_deux_points(centres[a_retirer])]
    sortie = binaire.copy()
    sortie[np.isin(etiquettes, a_retirer)] = 255
    return sortie


def nettoyer_bords(binaire, marge=PARAMS_DEFAUT["marge_bord"]):
    """Supprime les traces noires de bord (bord de photocopie ou de scan) : composantes entièrement dans une
    bande de marge (fraction de la plus petite dimension) qui touchent le bord de l'image ou forment un
    long trait fin. Le texte de marge (en-tête d'impression), fait de petits caractères, est conservé."""
    h, w = binaire.shape
    m = max(2, int(marge * min(h, w)))
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(255 - binaire, connectivity=8)
    x, y, l, ht = (stats[:, k] for k in range(4))
    touche = (x == 0) | (y == 0) | (x + l >= w) | (y + ht >= h)
    dans_marge = (x + l <= m) | (y + ht <= m) | (x >= w - m) | (y >= h - m)
    trait = ((l >= 0.05 * w) & (l >= 8 * ht)) | ((ht >= 0.05 * h) & (ht >= 8 * l))
    a_retirer = np.where(dans_marge & (touche | trait))[0]
    sortie = binaire.copy()
    sortie[np.isin(etiquettes, a_retirer[a_retirer > 0])] = 255
    return sortie


def normaliser(gris, params=None):
    """Enchaîne débruitage, retrait du fond, binarisation locale, remplissage des aplats, suppression
    des points isolés et des traces de bord. Renvoie (gris normalisé, binaire, paramètres)."""
    p = {**PARAMS_DEFAUT, **(params or {})}
    propre = retirer_fond(debruiter(gris, p["debruitage_h"]), dilatation=p["fond_dilatation"],
                          mediane=p["fond_mediane"])
    binaire = remplir_aplats(binariser(propre, p["bloc_binarisation"], p["constante_binarisation"]), propre,
                             p["seuil_aplat"])
    binaire = nettoyer_bords(retirer_points(binaire, p["aire_point_max"], p["rayon_isolement"]), p["marge_bord"])
    return propre, binaire, p
