"""Chaîne complète sur un petit corpus synthétique : scans d'origine inchangés (empreinte), sorties
attendues, journal complet, refus d'écrire dans le dossier des données."""
import csv

import cv2
import pymupdf
import pytest

from bdc.fichiers import empreinte, verifier_sortie
from bdc.journal import lire_journal
from bdc.orientation import tourner
from bdc.pdf_io import pages_pdf
from bdc.planche import planche_contact
from inventory import inventorier
from preprocess import pretraiter_dossier, pretraiter_page
from run_all import faire_planche, resumer


def ecrire_pdf(chemin, images):
    """PDF d'une page par image (JPEG embarqué, page A4), comme les bons de commande du corpus."""
    doc = pymupdf.open()
    for image in images:
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=cv2.imencode(".jpg", image)[1].tobytes())
    doc.save(chemin)


@pytest.fixture
def corpus(tmp_path, page):
    donnees = tmp_path / "data" / "BDC"
    donnees.mkdir(parents=True)
    ecrire_pdf(donnees / "BDC_90_droit.pdf", [page])
    ecrire_pdf(donnees / "BDC_91_envers.pdf", [tourner(page, 180), page])
    return donnees, tmp_path / "out"


def test_scans_inchanges_et_sorties(corpus):
    donnees, sortie = corpus
    avant = {p.name: empreinte(p) for p in donnees.iterdir()}
    lignes = inventorier(donnees, sortie)
    entrees = pretraiter_dossier(donnees, sortie)
    assert {p.name: empreinte(p) for p in donnees.iterdir()} == avant
    assert len(lignes) == 3 and len(entrees) == 3
    assert (sortie / "inventory.csv").exists() and len(list(csv.DictReader(open(sortie / "inventory.csv")))) == 3
    assert len(list((sortie / "pages").glob("*.png"))) == 3
    assert len(list((sortie / "preprocessed").glob("*.png"))) == 6
    journal = lire_journal(sortie / "preprocess_log.jsonl")
    assert [e["angle_orientation"] for e in journal] == [0, 180, 0]
    cles = {"fichier_source", "page", "angle_orientation", "angle_redressement", "perspective", "normalisation", "duree_s"}
    assert cles | {"diagnostic", "qualite", "empreinte"} <= set(journal[0])
    assert faire_planche(sortie, entrees).exists()
    assert "Pages traitées : 3" in resumer(entrees, 1.0)


def test_inventaire_resolution(corpus):
    donnees, _ = corpus
    page = pages_pdf(donnees / "BDC_90_droit.pdf")[0]
    assert round(page["dpi_x"]) == 150 and round(page["dpi_y"]) == 150


def test_refus_d_ecrire_dans_les_donnees(corpus):
    donnees, _ = corpus
    with pytest.raises(ValueError):
        verifier_sortie(donnees, donnees / "out")
    with pytest.raises(ValueError):
        inventorier(donnees, donnees)


def test_pretraiter_page_signale_les_alertes(page):
    _, binaire, infos = pretraiter_page(page)
    assert infos["perspective"] == "non necessaire" and infos["alertes"] == []
    assert binaire.ndim == 2


def test_planche_contact(page):
    planche = planche_contact([(page, page, "a"), (page, page, "b")])
    assert planche.ndim == 2 and planche.shape[0] > 0


def test_pages_depuis_octets_pdf_image_et_rendu(page, tmp_path):
    """Nouveau document : PDF scanné (image native), image seule, PDF généré (rendu), format refusé."""
    from bdc.pdf_io import pages_depuis_octets
    ecrire_pdf(tmp_path / "scan.pdf", [page, page])
    pages = pages_depuis_octets((tmp_path / "scan.pdf").read_bytes(), "scan.pdf")
    assert len(pages) == 2 and pages[0]["mode"] == "image native"
    image = pages_depuis_octets(cv2.imencode(".png", page)[1].tobytes(), "photo.png")
    assert image[0]["image"].shape == page.shape and image[0]["dpi_x"] is None
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Bon de commande genere par logiciel")
    rendu = pages_depuis_octets(doc.tobytes(), "genere.pdf")
    assert rendu[0]["mode"].startswith("rendu") and rendu[0]["image"].ndim == 2
    with pytest.raises(ValueError):
        pages_depuis_octets(b"abc", "commande.docx")
