"""Normalisation : tache atténuée, binaire propre, ponctuation conservée, bruit et bords retirés."""
import cv2
import numpy as np

from bdc.normalisation import (binariser, debruiter, estimer_fond, nettoyer_bords, normaliser, remplir_aplats,
                               retirer_fond, retirer_points)


def avec_tache(page):
    """Ajoute une tache de café (disque gris flou) au milieu de la page."""
    tache = np.zeros(page.shape, np.float32)
    cv2.circle(tache, (620, 1300), 160, 1.0, -1)
    tache = cv2.GaussianBlur(tache, (0, 0), 25)
    return (page.astype(np.float32) * (1 - 0.35 * tache)).astype(np.uint8)


def test_tache_attenuee(page):
    """Sous la tache, le binaire est le même que sur la page propre (à 1 % des pixels près)."""
    _, propre, _ = normaliser(page)
    _, tache, _ = normaliser(avec_tache(page))
    zone = (slice(1150, 1450), slice(470, 770))
    assert (propre[zone] != tache[zone]).mean() < 0.01


def test_binaire_noir_et_blanc(page):
    propre, binaire, params = normaliser(page)
    assert set(np.unique(binaire)) <= {0, 255}
    assert propre.shape == binaire.shape == page.shape
    assert params["bloc_binarisation"] == 31


def test_fond_estime_ignore_l_encre(page):
    assert estimer_fond(page).min() > 200
    assert retirer_fond(page).dtype == np.uint8


def test_debruiter_conserve_la_taille(page):
    assert debruiter(page).shape == page.shape


def test_binariser_et_remplir_aplats():
    gris = np.full((200, 400), 240, np.uint8)
    gris[50:150, 50:350] = 20  # bandeau noir
    binaire = binariser(gris)
    assert (binaire[90:110, 150:250] == 255).any()          # le seuil local évide l'aplat
    assert (remplir_aplats(binaire, gris)[90:110, 150:250] == 0).all()


def test_retirer_points_garde_deux_points_et_texte(page):
    sale = page.copy()
    sale[40, 40] = sale[60, 300] = 0                        # points de bruit isolés
    sale[1000:1002, 1150:1152] = sale[1010:1012, 1150:1152] = 0   # un deux-points isolé
    propre = retirer_points(sale)
    assert propre[40, 40] == 255 and propre[60, 300] == 255
    assert propre[1000, 1150] == 0 and propre[1010, 1150] == 0
    assert (propre[100:200] == page[100:200]).all()          # le texte n'est pas touché


def test_nettoyer_bords_retire_trait_de_bord(page):
    sale = page.copy()
    sale[3:6, 200:1000] = 0                                  # trait de bord de photocopie
    propre = nettoyer_bords(sale)
    assert (propre[3:6, 200:1000] == 255).all()
    assert (propre[100:900] == page[100:900]).all()
