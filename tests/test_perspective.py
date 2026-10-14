"""Perspective : feuille synthétique photographiée de biais sur un fond sombre."""
import cv2
import numpy as np

from bdc.perspective import corriger_perspective, detecter_coins, est_photo, ordonner_coins

COINS_PHOTO = np.float32([[260, 180], [1150, 230], [1210, 1600], [150, 1540]])


def photo_de(page, fond=60):
    """Simule une photo : la page est projetée en quadrilatère sur un fond sombre de 1400 x 1800."""
    h, w = page.shape
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), COINS_PHOTO)
    feuille = cv2.warpPerspective(page, m, (1400, 1800), borderValue=0)
    masque = cv2.warpPerspective(np.full_like(page, 255), m, (1400, 1800), borderValue=0)
    return np.where(masque > 0, feuille, fond).astype(np.uint8)


def test_photo_corrigee(page):
    image, statut, details = corriger_perspective(photo_de(page))
    assert statut == "appliquee" and details["format"] == "A4 impose"
    assert np.abs(np.array(details["coins"]) - ordonner_coins(COINS_PHOTO)).max() < 15
    assert abs(image.shape[0] / image.shape[1] - page.shape[0] / page.shape[1]) < 0.05


def test_scan_non_deforme(page):
    image, statut, details = corriger_perspective(page)
    assert statut == "non necessaire" and details is None and image is page


def test_echec_signale_sans_deformer():
    """Photo sans feuille rectangulaire (un disque clair) : rien n'est déformé, l'échec est signalé."""
    image = np.full((1800, 1400), 50, np.uint8)
    cv2.circle(image, (700, 900), 550, 230, -1)
    sortie, statut, _ = corriger_perspective(image)
    assert statut == "echec" and sortie is image


def test_detecter_coins_et_est_photo(page):
    assert est_photo(photo_de(page)) and not est_photo(page)
    assert detecter_coins(photo_de(page)) is not None


def test_ordonner_coins():
    melanges = np.float32([[10, 100], [100, 0], [0, 0], [100, 100]])
    assert ordonner_coins(melanges).tolist() == [[0, 0], [100, 0], [100, 100], [10, 100]]
