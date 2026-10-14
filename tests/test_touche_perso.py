"""Diagnostic du défaut, score de qualité et doublons, sur des pages synthétiques."""
import cv2
import numpy as np

from bdc.diagnostic import classer, diagnostiquer
from bdc.doublons import comparer, distance, empreinte, paires_proches
from bdc.normalisation import normaliser
from bdc.qualite import evaluer_qualite, rampe, verdict
from bdc.redressement import redresser
from test_perspective import photo_de


def fax_de(page):
    """Fax : image binaire, noir pur, semée de points de bruit."""
    rng = np.random.default_rng(0)
    fax = np.where(page < 128, 0, 255).astype(np.uint8)
    fax[rng.random(page.shape) < 0.004] = 0
    return fax


def test_diagnostic_des_defauts(page):
    assert diagnostiquer(page)[0] == "propre"
    assert diagnostiquer(photo_de(page))[0] == "photo"
    assert diagnostiquer(fax_de(page))[0] == "fax"
    pale = np.where(page < 128, 160, 255).astype(np.uint8)
    assert diagnostiquer(pale)[0] == "encre pale"
    teinte = np.where(page < 128, 120, 228).astype(np.uint8)
    assert diagnostiquer(teinte)[0] == "papier teinte"


def test_diagnostic_reglages_adaptes(page):
    defaut, reglages, indices = diagnostiquer(np.where(page < 128, 160, 255).astype(np.uint8))
    assert defaut == "encre pale" and reglages == {"constante_binarisation": 10}
    assert {"photo", "noir_p1", "extremes", "fond_median", "points", "aire_creux"} <= set(indices)


def test_classer_ordre_des_regles():
    base = dict(photo=False, noir_p1=50, extremes=0.9, fond_median=255, points=0.1, aire_creux=0.0)
    assert classer(base) == "propre"
    assert classer({**base, "aire_creux": 0.02}) == "tache ou ombre"
    assert classer({**base, "photo": True, "fond_median": 200}) == "photo"


def test_qualite_page_propre_prete(page):
    propre, binaire, _ = normaliser(page)
    q = evaluer_qualite(propre, binaire, 150, 1.0)
    assert q["verdict"] == "pret pour OCR" and q["score"] >= 80 and q["raisons"] == []


def test_qualite_page_degradee_signalee(page):
    pale = cv2.GaussianBlur(np.where(page < 128, 205, 255).astype(np.uint8), (0, 0), 1.5)
    propre, binaire, _ = normaliser(pale)
    q = evaluer_qualite(propre, binaire, 90, 0.05)
    assert q["verdict"] == "a rescanner"
    assert "résolution faible" in q["raisons"] and "orientation incertaine" in q["raisons"]


def test_rampe_et_verdict():
    assert rampe(125, 100, 150) == 0.5 and rampe(None, 0, 1) is None and rampe(300, 215, 150) == 0.0
    assert verdict(90, {"a": 1.0, "b": None}) == "pret pour OCR"
    assert verdict(90, {"a": 1.0, "b": 0.3}) == "a verifier"
    assert verdict(90, {"a": 1.0, "b": 0.0}) == "a rescanner"


def autre_bon(page):
    """Même formulaire, autres valeurs : trois lignes remplacées par des valeurs plus courtes, comme des
    champs remplis différemment (la mesure compare la place de l'encre, pas les lettres)."""
    autre = page.copy()
    autre[120:250] = 255
    for valeur, y in (("ZK 17", 150), ("0-30 cm", 192), ("EARL Pontreau", 234)):
        cv2.putText(autre, valeur, (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    return autre


def test_doublon_rescanne_detecte(page):
    rescan = cv2.resize(redresser(page, 0.4), None, fx=0.93, fy=0.93)
    niveau, d, ecart = comparer(page, rescan)
    assert niveau == "quasi identique" and ecart < 0.015


def test_document_different_jamais_pris_pour_doublon(page):
    """Mêmes en-tête et tableau, valeurs différentes : jamais « quasi identique » (au plus « même modèle »)."""
    niveau, _, ecart = comparer(page, autre_bon(page))
    assert niveau != "quasi identique"
    assert ecart is None or ecart > 0.015


def test_pages_differentes(page):
    vide = np.full_like(page, 255)
    cv2.putText(vide, "AUTRE DOCUMENT", (100, 900), cv2.FONT_HERSHEY_SIMPLEX, 4, 0, 8)
    assert comparer(page, vide)[0] is None


def test_empreinte_et_paires(page):
    e = empreinte(page)
    assert len(e) == 16 and distance(e, e) == 0
    paires = paires_proches({"a": page, "b": cv2.resize(redresser(page, 0.3), None, fx=0.95, fy=0.95)})
    assert paires and paires[0][2] == "quasi identique"


def test_niveaux_de_ressemblance(monkeypatch, page):
    """Le niveau dépend de l'écart de contenu : <= 1,5 % quasi identique, <= 4 % à vérifier, sinon même modèle."""
    import bdc.doublons as d
    for ecart, attendu in [(0.01, "quasi identique"), (0.03, "a verifier"), (0.08, "meme modele"), (None, "meme modele")]:
        monkeypatch.setattr(d, "ecart_contenu", lambda a, b, e=ecart: e)
        assert d.comparer(page, page)[0] == attendu
