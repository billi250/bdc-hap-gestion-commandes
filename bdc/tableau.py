"""Détection d'un tableau à grille (traits horizontaux et verticaux) et découpage en cellules.

Les longs traits sont isolés par ouverture morphologique (noyau horizontal, puis vertical) ; leur réunion
dessine la grille. Le plus grand bloc de grille ayant au moins 3 lignes et 3 colonnes est retenu ; les
positions des traits donnent les bornes des lignes et des colonnes du tableau.
"""
import cv2
import numpy as np

MIN_LIGNES, MIN_COLONNES = 3, 3


def traits(binaire, fraction=30):
    """Masques des longs traits horizontaux et verticaux (1 = trait)."""
    encre = (binaire == 0).astype(np.uint8)
    h, w = encre.shape
    horiz = cv2.morphologyEx(encre, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // fraction, 10), 1)))
    vert = cv2.morphologyEx(encre, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // fraction, 10))))
    return horiz, vert


def _positions(profil, seuil, ecart_min=6):
    """Centres des pics d'un profil (positions des traits), en fusionnant les traits voisins."""
    actifs = np.where(profil >= seuil)[0]
    if not len(actifs):
        return []
    groupes, debut = [], actifs[0]
    for a, b in zip(actifs, actifs[1:]):
        if b - a > ecart_min:
            groupes.append((debut + a) // 2)
            debut = b
    groupes.append((debut + actifs[-1]) // 2)
    return [int(g) for g in groupes]


def detecter_tableau(binaire):
    """Renvoie (rect, ys, xs) du plus grand tableau : rect = (x, y, l, h), ys et xs = positions des traits
    horizontaux et verticaux dans l'image. None si aucun tableau n'est trouvé."""
    horiz, vert = traits(binaire)
    grille = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(grille, connectivity=8)
    for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        x, y, l, h = stats[i, :4]
        zone_h, zone_v = horiz[y:y + h, x:x + l], vert[y:y + h, x:x + l]
        ys = _positions(zone_h.sum(axis=1), 0.5 * l)
        xs = _positions(zone_v.sum(axis=0), 0.5 * h)
        if len(ys) - 1 >= MIN_LIGNES and len(xs) - 1 >= MIN_COLONNES:
            return (int(x), int(y), int(l), int(h)), [y + v for v in ys], [x + v for v in xs]
    return None


def cellules(ys, xs, marge=3):
    """Rectangles (x0, y0, x1, y1) de chaque cellule, ligne par ligne, sans les traits."""
    return [[(xs[j] + marge, ys[i] + marge, xs[j + 1] - marge, ys[i + 1] - marge) for j in range(len(xs) - 1)]
            for i in range(len(ys) - 1)]
