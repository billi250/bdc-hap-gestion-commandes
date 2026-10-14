"""Brique 3 (première version) : extraction des champs et du tableau des échantillons, par OCR local.

Usage : python extract.py out/
À lancer après run_all.py (lit out/preprocessed/). Écrit out/extraction/<pdf>.json (un fichier par document),
out/extraction/champs.csv (une ligne par document) et out/extraction/echantillons.csv (une ligne par échantillon).
Nécessite Tesseract avec le français (macOS : brew install tesseract tesseract-lang).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from bdc import vlm
from bdc.extraction import extraire_document
from bdc.ocr import disponible

CHAMPS_CSV = ["emetteur", "emetteur_adresse", "emetteur_siret", "emetteur_telephone", "contact_emetteur",
              "client_final", "adresse_site", "parcelles", "commune", "client_telephone", "client_interlocuteur",
              "laboratoire", "numero_commande", "date_commande", "date_prelevement", "delai"]


def pages_par_document(dossier_pretraite):
    """{nom du PDF sans extension: [(gris, binaire) page par page]} à partir de out/preprocessed."""
    documents = defaultdict(list)
    for binaire in sorted(Path(dossier_pretraite).glob("*_p[0-9]*.png")):
        if binaire.stem.endswith("_gris"):
            continue
        nom, page = binaire.stem.rsplit("_p", 1)
        documents[nom].append((int(page), binaire))
    return {nom: [(cv2.imread(str(b.with_name(b.stem + "_gris.png")), cv2.IMREAD_GRAYSCALE),
                   cv2.imread(str(b), cv2.IMREAD_GRAYSCALE)) for _, b in sorted(pages)]
            for nom, pages in documents.items()}


def ligne_champs(nom, doc):
    """Une ligne du fichier champs.csv : valeur et confiance de chaque champ, nombre d'échantillons, alertes."""
    ligne = {"document": nom}
    for champ in CHAMPS_CSV:
        valeur = doc["champs"].get(champ) or {}
        ligne[champ], ligne[f"{champ}_confiance"] = valeur.get("valeur", ""), valeur.get("confiance", "")
    ligne["emails"] = " ; ".join(m["valeur"] for m in doc["champs"].get("emails") or [])
    ligne["nb_echantillons"] = len(doc["echantillons"])
    ligne["nb_annules"] = sum(e.get("statut") == "annule" for e in doc["echantillons"])
    ligne["confiance_tableau"] = doc["confiance_tableau"]
    ligne["alertes"] = " | ".join(doc["alertes"])
    return ligne


def extraire_dossier(dossier_sortie, moteur="tesseract", seulement=None):
    """Extrait les documents prétraités (tous, ou ceux dont le nom contient « seulement »). Renvoie {nom: document}."""
    sortie = Path(dossier_sortie) / "extraction"
    sortie.mkdir(parents=True, exist_ok=True)
    documents = {}
    for nom, pages in pages_par_document(Path(dossier_sortie) / "preprocessed").items():
        if seulement and seulement not in nom:
            continue
        doc = extraire_document(pages, moteur)
        (sortie / f"{nom}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        documents[nom] = doc
    lignes = [ligne_champs(n, d) for n, d in documents.items()]
    if lignes:
        with open(sortie / "champs.csv", "w", newline="", encoding="utf-8") as fh:
            ecrivain = csv.DictWriter(fh, fieldnames=list(lignes[0]))
            ecrivain.writeheader()
            ecrivain.writerows(lignes)
    echantillons = [{"document": n, **e} for n, d in documents.items() for e in d["echantillons"]]
    colonnes = list(dict.fromkeys(k for e in echantillons for k in e))
    with open(sortie / "echantillons.csv", "w", newline="", encoding="utf-8") as fh:
        ecrivain = csv.DictWriter(fh, fieldnames=colonnes or ["document"])
        ecrivain.writeheader()
        ecrivain.writerows(echantillons)
    return documents


def main():
    parser = argparse.ArgumentParser(description="Extraction des champs et des échantillons (brique 3, OCR local).")
    parser.add_argument("sortie", help="dossier de sortie de run_all.py (ex : out/)")
    parser.add_argument("--moteur", choices=["tesseract", "vlm", "les deux"], default="tesseract",
                        help="tesseract (rapide), vlm (Qwen3-VL local, environ 30 s par page) ou les deux (comparés)")
    parser.add_argument("--document", help="ne traiter que les documents dont le nom contient ce texte")
    args = parser.parse_args()
    if args.moteur != "vlm" and not disponible():
        sys.exit("Tesseract (avec le français) est introuvable : brew install tesseract tesseract-lang")
    if args.moteur != "tesseract" and not vlm.disponible():
        sys.exit(f"Modèle {vlm.MODELE} indisponible : lancer Ollama puis « ollama pull {vlm.MODELE} »")
    documents = extraire_dossier(args.sortie, args.moteur, args.document)
    print(f"{len(documents)} documents extraits, {sum(len(d['echantillons']) for d in documents.values())} "
          f"échantillons, {sum(bool(d['alertes']) for d in documents.values())} documents avec alerte.")


if __name__ == "__main__":
    main()
