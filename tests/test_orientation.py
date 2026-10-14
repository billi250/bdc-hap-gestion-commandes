"""Orientation : on fait pivoter nous-mêmes une page et on vérifie la correction détectée."""
import numpy as np
import pytest

from bdc.orientation import (detecter_axe, detecter_haut_bas, detecter_orientation, longueur_lignes,
                             texte_seul, tourner)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_detecter_orientation_retrouve_la_rotation(page, rotation):
    """Une page tournée de r degrés (sens horaire) doit être corrigée de (360 - r) degrés."""
    angle, confiance, _ = detecter_orientation(tourner(page, rotation))
    assert angle == (360 - rotation) % 360
    assert confiance > 0


def test_tourner_quatre_quarts_revient_au_depart(page):
    assert np.array_equal(tourner(tourner(page, 90), 270), page)
    assert tourner(page, 90).shape == page.shape[::-1]


def test_detecter_axe(page):
    assert detecter_axe(page)[0] == 0
    assert detecter_axe(tourner(page, 90))[0] == 90


def test_detecter_haut_bas(page):
    assert detecter_haut_bas(page)[0] == 0
    assert detecter_haut_bas(tourner(page, 180))[0] == 180


def test_longueur_lignes_plus_grande_a_l_horizontale(page):
    texte = texte_seul(page)
    assert longueur_lignes(texte) > 5 * longueur_lignes(tourner(texte, 90))
