"""Redressement : on incline une page d'un angle connu et on vérifie l'angle estimé."""
import pytest

from bdc.redressement import estimer_inclinaison, redresser, score_horizontalite
from bdc.encre import masque_encre


@pytest.mark.parametrize("inclinaison", [2.0, -2.0, 1.3, -2.5])
def test_estimer_inclinaison_angle_connu(page, inclinaison):
    """Page tournée de +2° : il faut appliquer -2° pour la remettre à plat (tolérance 0,15°)."""
    angle, _ = estimer_inclinaison(redresser(page, inclinaison))
    assert angle == pytest.approx(-inclinaison, abs=0.15)


def test_page_droite_reste_droite(page):
    assert abs(estimer_inclinaison(page)[0]) <= 0.1


def test_redresser_agrandit_le_cadre_sans_couper(page):
    tournee = redresser(page, 2.0)
    assert tournee.shape[0] > page.shape[0] and tournee.shape[1] > page.shape[1]
    assert redresser(page, 0.0) is page


def test_score_maximal_a_plat(page):
    encre = masque_encre(page)
    assert score_horizontalite(encre, 0.0) > score_horizontalite(encre, 2.0)
