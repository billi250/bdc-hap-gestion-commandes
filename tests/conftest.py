"""Pages synthétiques pour les tests : aucune donnée réelle n'est utilisée (les scans ne sont pas versionnés)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LIGNES = [
    "BON DE COMMANDE D'ANALYSES - Laboratoire des sols",
    "Date de prelevement : 21/09/2026 - Parcelle ZK 42",
    "Demandeur : Cooperative agricole du Val de Brenne",
    "Analyses demandees : HAP 16, matiere seche, pH",
    "Profondeur : 0-25 cm ; conditionnement flacon verre",
    "Le laboratoire doit recevoir les echantillons sous 24 h",
    "Facturation : compte client 41-0087, delai standard",
    "Remarques : bord de route, ancien depot de machefers",
]


def page_synthetique(hauteur=1754, largeur=1240):
    """Page A4 à 150 dpi : un titre, des lignes de texte et un tableau, noir sur blanc."""
    page = np.full((hauteur, largeur), 255, np.uint8)
    y = 150
    for i in range(3):
        for texte in LIGNES:
            cv2.putText(page, texte, (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
            y += 42
        y += 30
    cv2.rectangle(page, (90, y), (largeur - 90, y + 200), 0, 2)
    for k in range(1, 4):
        cv2.line(page, (90, y + 50 * k), (largeur - 90, y + 50 * k), 0, 2)
    return page


@pytest.fixture
def page():
    return page_synthetique()
