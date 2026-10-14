"""Lien entre un échantillon du BDC et ses résultats HAP du projet GC-MS.

Le projet GC-MS exporte ses résultats en CSV (out/gcms_resultats.csv, jamais versionné), une ligne par
échantillon et par composé, avec au moins les colonnes : echantillon, compose, concentration, unite.
Le séparateur (« ; » ou « , ») est détecté. Sans ce fichier, le lien est simplement indisponible.
"""
import csv
from pathlib import Path

from .champs import sans_accents


def _cle(texte):
    return "".join(c for c in sans_accents(texte or "") if c.isalnum())


def lire(chemin):
    """Lignes du CSV des résultats (liste de dict) ; liste vide si le fichier n'existe pas."""
    chemin = Path(chemin) if chemin else None
    if not chemin or not chemin.exists():
        return []
    texte = chemin.read_text(encoding="utf-8-sig")
    separateur = ";" if texte.split("\n", 1)[0].count(";") >= texte.split("\n", 1)[0].count(",") else ","
    return list(csv.DictReader(texte.splitlines(), delimiter=separateur))


def resultats(reference, chemin):
    """Résultats d'un échantillon : correspondance exacte sur la référence (sans ponctuation ni casse)."""
    lignes = lire(chemin)
    if not lignes:
        return dict(disponible=False, message="résultats GC-MS non importés (out/gcms_resultats.csv absent)")
    cle = _cle(reference)
    trouves = [l for l in lignes if _cle(l.get("echantillon")) == cle]
    return dict(disponible=True, reference=reference, resultats=trouves,
                message=None if trouves else "aucun résultat GC-MS pour cet échantillon")
