"""Planche contact : pour chaque PDF, la page 1 avant et après prétraitement, côte à côte."""
import cv2
import numpy as np

HAUTEUR = 420       # hauteur d'une vignette (pixels)
PAIRES_PAR_LIGNE = 4


def vignette(image, hauteur=HAUTEUR):
    """Réduit une image à la hauteur donnée (niveaux de gris)."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = hauteur / image.shape[0]
    return cv2.resize(image, (max(1, int(image.shape[1] * f)), hauteur), interpolation=cv2.INTER_AREA)


def paire(avant, apres, titre, largeur_case):
    """Case de la planche : titre au-dessus, avant à gauche, après à droite, fond gris clair."""
    a, b = vignette(avant), vignette(apres)
    case = np.full((HAUTEUR + 40, largeur_case), 200, np.uint8)
    case[35:35 + HAUTEUR, 5:5 + a.shape[1]] = a
    x = min(15 + a.shape[1], largeur_case - b.shape[1])
    case[35:35 + HAUTEUR, x:x + b.shape[1]] = b
    cv2.putText(case, titre, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    return case


def planche_contact(paires):
    """Assemble les cases [(avant, après, titre)] en une grille de PAIRES_PAR_LIGNE colonnes."""
    largeur = max(vignette(a).shape[1] + vignette(b).shape[1] for a, b, _ in paires) + 25
    cases = [paire(a, b, t, largeur) for a, b, t in paires]
    vide = np.full_like(cases[0], 255)
    cases += [vide] * (-len(cases) % PAIRES_PAR_LIGNE)
    lignes = [np.hstack(cases[i:i + PAIRES_PAR_LIGNE]) for i in range(0, len(cases), PAIRES_PAR_LIGNE)]
    return np.vstack(lignes)
