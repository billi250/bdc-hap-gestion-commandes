"""Détection et correction de l'orientation (0, 90, 180, 270 degrés) sans OCR.

Méthode (vote ligne par ligne, à la résolution native) :
1. Lignes de texte : on retire les longs traits (cadres, tableaux), puis une fermeture horizontale
   relie les caractères d'une même ligne en un bloc large et peu haut. Sur une page tournée de 90°,
   les mêmes blocs sont hauts et étroits : on retient l'axe qui donne la plus grande longueur cumulée
   de blocs « ligne de texte ».
2. Haut ou bas : en écriture latine, les jambages montants (b, d, f, h, k, l, t, majuscules, accents)
   sont plus fréquents que les descendants (g, j, p, q, y). Dans chaque ligne, on compare l'encre
   au-dessus et au-dessous du coeur de la ligne (zone des minuscules) ; chaque ligne vote, la majorité
   pondérée tranche entre 0 et 180 (ou 90 et 270).
"""
import cv2
import numpy as np

from .encre import masque_encre, retirer_traits

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
COTE_ORIENTATION = 1800    # résolution de travail (environ la résolution native du corpus, 150 dpi)
HAUTEUR_LIGNE = (8, 45)    # hauteur plausible d'une ligne de texte (pixels, à cette résolution)
FERMETURE = 15             # largeur de la fermeture horizontale qui relie les caractères (pixels)
SEUIL_CONFIANCE = 0.25     # vote net (haut - bas) / total en dessous duquel l'orientation est incertaine


def tourner(image, angle):
    """Tourne l'image de angle degrés dans le sens horaire (0, 90, 180 ou 270)."""
    code = ROTATIONS[angle % 360]
    return image if code is None else cv2.rotate(image, code)


def blocs_lignes(texte):
    """Rectangles (x, y, l, h) des blocs « ligne de texte » : caractères reliés horizontalement,
    de hauteur plausible et nettement plus larges que hauts."""
    relie = cv2.morphologyEx(texte, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (FERMETURE, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(relie, connectivity=8)
    return [tuple(s[:4]) for s in stats[1:]
            if HAUTEUR_LIGNE[0] <= s[3] <= HAUTEUR_LIGNE[1] and s[2] >= 4 * s[3]]


def longueur_lignes(texte):
    """Longueur cumulée des blocs « ligne de texte » : grande si le texte est horizontal."""
    return int(sum(l for _, _, l, _ in blocs_lignes(texte)))


def vote_ligne(texte, bloc):
    """Vote d'une ligne : (encre au-dessus du coeur - encre au-dessous) / total, entre -1 et 1."""
    x, y, l, h = bloc
    profil = texte[y:y + h, x:x + l].sum(axis=1).astype(np.float64)
    coeur = np.where(profil >= 0.5 * profil.max())[0]
    haut, bas = profil[:coeur[0]].sum(), profil[coeur[-1] + 1:].sum()
    return (haut - bas) / (haut + bas) if haut + bas else 0.0


def score_haut_bas(texte):
    """Vote pondéré de toutes les lignes : > 0 si le texte est à l'endroit, < 0 s'il est à l'envers.
    Chaque ligne pèse sa largeur (les longues lignes de texte comptent plus que les fragments)."""
    blocs = blocs_lignes(texte)
    poids = np.array([b[2] for b in blocs], dtype=np.float64)
    if not len(blocs):
        return 0.0
    votes = np.sign([vote_ligne(texte, b) for b in blocs])
    return float((votes * poids).sum() / poids.sum())


def texte_seul(gris):
    """Masque d'encre à la résolution de travail, sans les longs traits."""
    return retirer_traits(masque_encre(gris, COTE_ORIENTATION))


def detecter_axe(gris):
    """Renvoie (angle, confiance, details) : 0 si les lignes de texte sont horizontales, 90 sinon
    (rotation horaire qui les rend horizontales ; le sens haut/bas reste à trancher)."""
    texte = texte_seul(gris)
    longueurs = {a: longueur_lignes(tourner(texte, a)) for a in (0, 90)}
    angle = 0 if longueurs[0] >= longueurs[90] else 90
    confiance = float(min(1.0, max(longueurs.values()) / (3 * min(longueurs.values()) + 1e-9)))
    return angle, confiance, dict(longueur_lignes_0=longueurs[0], longueur_lignes_90=longueurs[90])


def detecter_haut_bas(gris):
    """Sur une page aux lignes horizontales et redressée : renvoie (angle, confiance, details),
    angle = 0 si la page est à l'endroit, 180 si elle est à l'envers."""
    score = score_haut_bas(texte_seul(gris))
    return (0 if score >= 0 else 180), float(min(1.0, abs(score) / SEUIL_CONFIANCE)), dict(vote_haut_bas=round(score, 3))


def detecter_orientation(gris):
    """Orientation complète d'une page : axe, redressement fin, puis haut/bas.
    Renvoie (angle, confiance, details) : angle horaire (0, 90, 180, 270) qui remet la page à l'endroit."""
    from .redressement import estimer_inclinaison, redresser
    axe, conf_axe, d_axe = detecter_axe(gris)
    a_plat = tourner(gris, axe)
    a_plat = redresser(a_plat, estimer_inclinaison(a_plat)[0])
    hb, conf_hb, d_hb = detecter_haut_bas(a_plat)
    return (axe + hb) % 360, min(conf_axe, conf_hb), {**d_axe, **d_hb}
