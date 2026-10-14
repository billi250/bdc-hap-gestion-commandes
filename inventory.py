"""Brique 0 : inventaire du corpus et rendu de chaque page en PNG.

Usage : python inventory.py data/BDC out/
Écrit out/inventory.csv (une ligne par page) et out/pages/<pdf>_p<n>.png (image native, sans rééchantillonnage).
"""
import argparse
import csv
from pathlib import Path

from bdc.fichiers import ecrire_png, empreinte, nom_page, verifier_sortie
from bdc.pdf_io import lister_pdf, pages_pdf

COLONNES = ["fichier", "nb_pages", "page", "largeur_px", "hauteur_px", "dpi_x", "dpi_y",
            "largeur_mm", "hauteur_mm", "canaux", "taille_octets", "sha256"]
MM_PAR_PT = 25.4 / 72


def decrire_page(chemin, nb_pages, page):
    """Une ligne d'inventaire pour une page."""
    img = page["image"]
    return dict(fichier=chemin.name, nb_pages=nb_pages, page=page["page"], largeur_px=img.shape[1],
                hauteur_px=img.shape[0], dpi_x=round(page["dpi_x"], 1), dpi_y=round(page["dpi_y"], 1),
                largeur_mm=round(page["largeur_pt"] * MM_PAR_PT, 1), hauteur_mm=round(page["hauteur_pt"] * MM_PAR_PT, 1),
                canaux=1 if img.ndim == 2 else img.shape[2], taille_octets=chemin.stat().st_size,
                sha256=empreinte(chemin))


def inventorier(dossier_donnees, dossier_sortie):
    """Inventorie tous les PDF, rend chaque page dans out/pages/ et écrit out/inventory.csv.
    Renvoie la liste des lignes d'inventaire."""
    verifier_sortie(dossier_donnees, dossier_sortie)
    sortie = Path(dossier_sortie)
    lignes = []
    for chemin in lister_pdf(dossier_donnees):
        pages = pages_pdf(chemin)
        for page in pages:
            lignes.append(decrire_page(chemin, len(pages), page))
            ecrire_png(sortie / "pages" / nom_page(chemin, page["page"]), page["image"])
    sortie.mkdir(parents=True, exist_ok=True)
    with open(sortie / "inventory.csv", "w", newline="", encoding="utf-8") as fh:
        ecrivain = csv.DictWriter(fh, fieldnames=COLONNES)
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    return lignes


def main():
    parser = argparse.ArgumentParser(description="Inventaire du corpus de bons de commande (brique 0).")
    parser.add_argument("donnees", help="dossier des PDF (ex : data/BDC)")
    parser.add_argument("sortie", help="dossier de sortie (ex : out/)")
    args = parser.parse_args()
    lignes = inventorier(args.donnees, args.sortie)
    print(f"{len({l['fichier'] for l in lignes})} PDF, {len(lignes)} pages inventoriées.")


if __name__ == "__main__":
    main()
