"""Reconstruction du tableau : ligne barrée, annulation par les observations, références, profondeur, analyses."""
import cv2
import numpy as np

from bdc.reconstruction import (analyses, annule_par_observation, ligne_barree, modele_references, profondeur,
                                reconstruire, reference)


def test_ligne_barree():
    ligne = np.full((60, 800), 255, np.uint8)
    cv2.putText(ligne, "ECO-4587-4   0-30   HAP 16 + MS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    assert not ligne_barree(ligne, 0, 60, 0, 800)
    cv2.line(ligne, (5, 30), (600, 27), 0, 2)
    assert ligne_barree(ligne, 0, 60, 0, 800)


def test_annule_par_observation_tolerant():
    assert annule_par_observation("annulé (flacon cassé)") and annule_par_observation("anndlé (lagon cncté)")
    assert not annule_par_observation("rang 40")


def test_modele_et_references():
    lues = ["ELO-4597-1", "€CO0-4587-2", "EC0-4587-3", "EC0-87-5"]
    modele = modele_references(lues, "004587")
    assert modele == ["ECO", "4587"]
    assert reference(modele, "LÉ", 4) == "ECO-4587-4" and reference(modele, "EC0-87-5", 5) == "ECO-4587-5"


def test_profondeur_et_analyses():
    assert profondeur("9-30") == "0-30" and profondeur("30-60") == "30-60" and profondeur("O-3O") == "0-30"
    assert analyses("HAP 16 +: MÇ") == ["HAP 16", "MS"] and analyses("HAP 16") == ["HAP 16"]


def test_reconstruire_bon_ecotellus():
    lignes = [dict(reference="ELO-4597-1", profondeur="9-30", analyses="HAP 16 + MS", remarques="rang 12", barree=False),
              dict(reference="€CO0-4587-2", profondeur="0-30", analyses="HAP 16 + MS", remarques="rang 40", barree=False),
              dict(reference="LÉ", profondeur=". .", analyses="his —p 16", remarques="anndlé (lagon cncté)", barree=True),
              dict(reference="EC0-4587-4", profondeur="0-30", analyses="HAP 16", remarques="tou EC0-45817-2", barree=False)]
    r = reconstruire(lignes, "004587", "ECOTELLUS")
    assert [e["reference"] for e in r] == ["ECO-4587-1", "ECO-4587-2", "ECO-4587-3", "ECO-4587-4"]
    assert [e["statut"] for e in r] == ["a analyser", "a analyser", "annule", "a analyser"]
    assert r[2]["motif_statut"] == "ligne barrée, confirmé par les observations"
    assert r[0]["profondeur"] == "0-30" and r[0]["corrections"]
    assert "ECO-4587-2" in r[3]["remarques"]
