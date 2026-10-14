"""Correction de perspective des photos de smartphone : 4 coins de la feuille, puis transformation.

Un scan remplit tout le cadre d'un fond blanc : il n'y a pas de bord de feuille à chercher. Une photo
montre la feuille claire sur un fond plus sombre (table, bureau). On repère donc d'abord si la page est
une photo (bordure nettement plus sombre que le centre), puis on cherche le plus grand contour clair à
4 sommets. Si les 4 coins ne sont pas trouvés, l'image n'est pas déformée et l'échec est signalé.
"""
import cv2
import numpy as np

from .encre import reduire

COTE_PERSPECTIVE = 800
ECART_BORDURE = 40        # écart de luminosité bordure / centre au-delà duquel on considère une photo
AIRE_MIN = 0.25           # la feuille doit couvrir au moins 25 % de la photo
AIRE_MAX = 0.97           # au-delà, le contour trouvé est le cadre de l'image, pas la feuille
RAPPORT_A4 = 2 ** 0.5     # 297 / 210
TOLERANCE_A4 = (1.2, 1.6) # rapports mesurés que l'on ramène au format A4


def est_photo(gris, bande=0.04):
    """Vrai si la bordure de l'image est nettement plus sombre que le centre (feuille posée sur un fond)."""
    h, w = gris.shape
    b = max(2, int(bande * min(h, w)))
    bordure = np.concatenate([gris[:b].ravel(), gris[-b:].ravel(), gris[:, :b].ravel(), gris[:, -b:].ravel()])
    centre = gris[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float(np.median(centre)) - float(np.median(bordure)) > ECART_BORDURE


def ordonner_coins(points):
    """Ordonne 4 points : haut-gauche, haut-droite, bas-droite, bas-gauche."""
    p = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s, d = p.sum(axis=1), np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)


def detecter_coins(gris):
    """Renvoie les 4 coins de la feuille (coordonnées de l'image d'origine, ordonnés) ou None."""
    petit, f = reduire(gris, COTE_PERSPECTIVE)
    flou = cv2.GaussianBlur(petit, (5, 5), 0)
    _, clair = cv2.threshold(flou, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clair = cv2.morphologyEx(clair, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(clair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    feuille = cv2.convexHull(max(contours, key=cv2.contourArea))
    aire = cv2.contourArea(feuille) / (petit.shape[0] * petit.shape[1])
    approx = cv2.approxPolyDP(feuille, 0.02 * cv2.arcLength(feuille, True), True)
    if len(approx) != 4 or not AIRE_MIN <= aire <= AIRE_MAX:
        return None
    return ordonner_coins(approx / f)


def dimensions_cible(coins):
    """Taille du rectangle de sortie. Les côtés mesurés sur la photo sont raccourcis par la perspective :
    si leur rapport est proche de celui d'un A4 (racine de 2, en portrait ou en paysage), on impose le
    rapport A4 en gardant le plus grand côté mesuré. Renvoie (largeur, hauteur, format)."""
    hg, hd, bd, bg = coins
    largeur = max(np.linalg.norm(hd - hg), np.linalg.norm(bd - bg))
    hauteur = max(np.linalg.norm(bg - hg), np.linalg.norm(bd - hd))
    rapport = max(largeur, hauteur) / min(largeur, hauteur)
    if TOLERANCE_A4[0] <= rapport <= TOLERANCE_A4[1]:
        if hauteur >= largeur:
            largeur = hauteur / RAPPORT_A4
        else:
            hauteur = largeur / RAPPORT_A4
        return int(round(largeur)), int(round(hauteur)), "A4 impose"
    return int(round(largeur)), int(round(hauteur)), "mesure"


def redresser_perspective(image, coins):
    """Projette le quadrilatère des coins sur un rectangle. Renvoie (image, format retenu)."""
    largeur, hauteur, format_ = dimensions_cible(coins)
    cible = np.array([[0, 0], [largeur - 1, 0], [largeur - 1, hauteur - 1], [0, hauteur - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(coins, cible)
    return cv2.warpPerspective(image, m, (largeur, hauteur), flags=cv2.INTER_CUBIC, borderValue=255), format_


def corriger_perspective(gris):
    """Renvoie (image, statut, details). statut : « appliquee », « non necessaire » (scan, pas de bord de
    feuille) ou « echec » (photo dont les 4 coins n'ont pas été trouvés : image laissée telle quelle).
    details : coins de la feuille et format de sortie (None si rien n'est appliqué)."""
    if not est_photo(gris):
        return gris, "non necessaire", None
    coins = detecter_coins(gris)
    if coins is None:
        return gris, "echec", None
    image, format_ = redresser_perspective(gris, coins)
    return image, "appliquee", dict(coins=coins.round(1).tolist(), format=format_)
