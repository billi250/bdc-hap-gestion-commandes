"""Extraction (brique 3) : règles des champs sur des lignes OCR simulées, grille de tableau, cases cochées,
et chaîne complète avec Tesseract sur une page synthétique (sautée si Tesseract est absent)."""
import cv2
import numpy as np
import pytest

from bdc import champs as c
from bdc.extraction import colonnes_de_marques, extraire_document, nom_colonne
from bdc.ocr import disponible
from bdc.tableau import cellules, detecter_tableau


def lignes(*textes, confiance=90.0):
    """Lignes OCR simulées (texte, confiance), de haut en bas."""
    return [dict(texte=t, confiance=confiance, boite=[0, 20 * i, 100, 15], mots=[]) for i, t in enumerate(textes)]


def test_luhn():
    assert c.luhn("12345678900015")          # numéro fabriqué avec une clé valide
    assert not c.luhn("12345678900016")


def test_numero_commande_formats():
    assert c.numero_commande(lignes("BON DE COMMANDE N° MST/2026/118"))["valeur"] == "MST/2026/118"
    assert c.numero_commande(lignes("Campagne 2026 - Réf. VB-26-0471"))["valeur"] == "VB-26-0471"
    assert c.numero_commande(lignes("N° 004587"))["valeur"] == "004587"
    assert c.numero_commande(lignes("N° AGS-2026-0527 - 27/05/2026"))["valeur"] == "AGS-2026-0527"


def test_dates_priorites_et_lettres():
    l = lignes("Date de prélèvement : 12/09/2026", "N° AGS-2026-0914 13/09/2026")
    assert c.date_commande(l)["valeur"] == "13/09/2026"
    assert c.date_prelevement(l)["valeur"] == "12/09/2026"
    assert c.date_commande(lignes("Saulnes-la-Côte, le 26 septembre 2026"))["valeur"] == "26/09/2026"
    assert c.date_commande(lignes("| 0-30 | 30/09/2026 | x")) is None     # date de tableau : écartée


def test_roles_emetteur_et_client_final():
    """Bon d'un bureau d'études : émetteur et son chargé d'affaires, client final et son site, laboratoire."""
    l = lignes("ECOTELLUS BON DE COMMANDE", "21 avenue des Platanes 33610 Cestas-les-Pins",
               "05 36 49 30 12 SIRET 794 120 358 00029",
               "Date : 30/09/2026 Chargé d'affaires : T. Renaudin",
               "Client final / site SCEA Clos des Aubiers (vignoble)",
               "Adresse du site : Route des Graves, 33580 St-Vernac",
               "Parcelle(s) : AK 77 / AK 78 Tél. : 06 39 98 20 64", "Laboratoire : Onenne Analycec")
    ch = c.extraire_champs(l)
    assert ch["emetteur"]["valeur"] == "ECOTELLUS"
    assert ch["emetteur_adresse"]["valeur"].startswith("21 avenue des Platanes")
    assert ch["contact_emetteur"]["valeur"] == "T. Renaudin"
    assert ch["client_final"]["valeur"] == "SCEA Clos des Aubiers (vignoble)"
    assert ch["parcelles"]["valeur"] == "AK 77 / AK 78"
    assert ch["emetteur_telephone"]["valeur"] == "05 36 49 30 12"
    assert ch["client_telephone"]["valeur"] == "06 39 98 20 64"
    assert ch["laboratoire"]["valeur"] == "ORVENNE ANALYSES" and ch["laboratoire"]["lue"] == "Onenne Analycec"
    assert ch["date_commande"]["valeur"] == "30/09/2026"


def test_etiquette_compte_client_ignoree():
    ch = c.extraire_champs(lignes("AGRISOL CONSEIL", "Facturation : compte client 41-0087",
                                  "Interlocuteur : Lucie Dumarais 06 39 98 52 71"))
    assert ch["client_final"] is None and ch["client_interlocuteur"]["valeur"] == "Lucie Dumarais"


def test_contacts_et_siret():
    l = lignes("Tél. 02 61 91 77 04 agronomie @valdebrenne-coop.fr", "SIRET 781 254 903 00026",
               "SIRET 781 254 903 00026")
    tels, mails = c.contacts(l)
    assert tels[0]["valeur"] == "02 61 91 77 04" and mails[0]["valeur"] == "agronomie@valdebrenne-coop.fr"
    s = c.sirets(l)
    assert len(s) == 1 and s[0]["valeur"] == "78125490300026" and s[0]["cle_valide"] is False


def test_emetteur_ignore_entete_de_fax():
    l = lignes("22/09/2026 15:40 DE: 03 53 01 77 21 A: 02 61 91 40 13 P.001/001", "MISSION SOLS & TERRITOIRES")
    assert c.emetteur(l)["valeur"] == "MISSION SOLS & TERRITOIRES"


def test_noms_de_colonnes_et_marques():
    noms = []
    for entete in ["N° échantillon client", "Profondeur (cm)", "HAP 16", "MS", "Remarques", "???"]:
        noms.append(nom_colonne(entete, noms))
    assert noms[:5] == ["reference", "profondeur", "hap16", "ms", "remarques"]
    rangs = colonnes_de_marques([{"inconnue": "X"}, {"inconnue": ""}, {"inconnue": "x"}], ["inconnue"])
    assert [r["inconnue"] for r in rangs] == [True, False, True]


def page_formulaire():
    """Bon synthétique : en-tête, champs étiquetés, délai à cocher, tableau de 3 échantillons."""
    page = np.full((1754, 1240), 255, np.uint8)
    ecrire = lambda t, x, y, s=0.9: cv2.putText(page, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, 0, 2, cv2.LINE_AA)
    ecrire("TERRES DU NORD", 90, 120, 1.4)
    ecrire("BON DE COMMANDE N° TN-2026-042", 90, 200)
    ecrire("Date de commande : 21/09/2026", 90, 260)
    ecrire("Raison sociale : EARL des Saules", 90, 320)
    ecrire("Tel. 03 21 45 67 89", 90, 380)
    for i, (mot, coche) in enumerate([("Standard", False), ("Urgent", True), ("Express", False)]):
        x = 300 + 260 * i
        cv2.rectangle(page, (x, 440), (x + 26, 466), 0, 2)
        if coche:
            cv2.line(page, (x + 5, 445), (x + 21, 461), 0, 3)
            cv2.line(page, (x + 21, 445), (x + 5, 461), 0, 3)
        ecrire(mot, x + 40, 465)
    ecrire("Delai :", 90, 465)
    xs, ys = [90, 400, 650, 850, 1150], [560 + 60 * k for k in range(5)]
    for y in ys:
        cv2.line(page, (xs[0], y), (xs[-1], y), 0, 2)
    for x in xs:
        cv2.line(page, (x, ys[0]), (x, ys[-1]), 0, 2)
    for j, t in enumerate(["Reference", "Profondeur", "HAP 16", "Remarques"]):
        ecrire(t, xs[j] + 10, ys[0] + 40, 0.8)
    for i, (ref, prof, hap) in enumerate([("TN-01", "0-30", True), ("TN-02", "30-60", False), ("TN-03", "0-30", True)]):
        y = ys[i + 1] + 40
        ecrire(ref, xs[0] + 10, y, 0.8)
        ecrire(prof, xs[1] + 10, y, 0.8)
        if hap:
            ecrire("X", xs[2] + 80, y, 0.9)
    return page


def test_detecter_tableau_et_cellules():
    trouve = detecter_tableau(page_formulaire())
    assert trouve is not None
    _, ys, xs = trouve
    assert len(ys) == 5 and len(xs) == 5
    assert len(cellules(ys, xs)) == 4 and len(cellules(ys, xs)[0]) == 4


@pytest.mark.skipif(not disponible(), reason="Tesseract (français) non installé")
def test_extraction_complete_page_synthetique():
    page = page_formulaire()
    doc = extraire_document([(page, page)])
    ch = doc["champs"]
    assert ch["numero_commande"]["valeur"] == "TN-2026-042"
    assert ch["date_commande"]["valeur"] == "21/09/2026"
    assert "Saules" in ch["client_final"]["valeur"]
    assert ch["delai"]["valeur"] == "urgent"
    refs = [e.get("reference") for e in doc["echantillons"]]
    assert refs == ["TN-01", "TN-02", "TN-03"]
    assert [e["hap16"] for e in doc["echantillons"]] == [True, False, True]
