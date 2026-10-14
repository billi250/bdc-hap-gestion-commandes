"""Brique 1 : prétraitement de chaque page (diagnostic, perspective, orientation, redressement, normalisation,
qualité) et recherche des doublons.

Usage : python preprocess.py data/BDC out/
Écrit out/preprocessed/<pdf>_p<n>.png (binaire), <pdf>_p<n>_gris.png (niveaux de gris normalisés),
out/preprocess_log.jsonl (un enregistrement par page) et out/doublons.csv. Les scans d'origine ne sont
jamais modifiés.

Ordre retenu : diagnostic du défaut sur l'image d'origine (il choisit les réglages de normalisation), puis
perspective (l'orientation se mesure mieux sur une feuille remise à plat), axe des lignes (0 ou 90°),
redressement fin, et seulement ensuite haut/bas (0 ou 180°) : le vote haut/bas est faussé par une inclinaison
de 1 à 3°, alors que le redressement ne dépend pas du sens. La qualité est évaluée sur le résultat.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2

from bdc import doublons
from bdc.diagnostic import diagnostiquer
from bdc.fichiers import ecrire_png, empreinte, nom_page, verifier_sortie
from bdc.journal import ecrire_entree, initialiser_journal
from bdc.normalisation import normaliser
from bdc.orientation import detecter_axe, detecter_haut_bas, tourner
from bdc.pdf_io import en_gris, lister_pdf, pages_pdf
from bdc.perspective import corriger_perspective
from bdc.qualite import evaluer_qualite
from bdc.redressement import estimer_inclinaison, redresser

SEUIL_CONFIANCE_ORIENTATION = 0.3   # en dessous, l'orientation est signalée comme incertaine


def orienter(gris):
    """Axe, redressement fin puis haut/bas. Renvoie (image à l'endroit et à plat, infos pour le journal)."""
    axe, conf_axe, d_axe = detecter_axe(gris)
    image = tourner(gris, axe)
    angle, d_red = estimer_inclinaison(image)
    image = redresser(image, angle)
    hb, conf_hb, d_hb = detecter_haut_bas(image)
    image = tourner(image, hb)
    confiance = round(min(conf_axe, conf_hb), 3)
    return image, dict(angle_orientation=(axe + hb) % 360, confiance_orientation=confiance,
                       details_orientation={**d_axe, **d_hb}, angle_redressement=angle, details_redressement=d_red)


def alertes(infos):
    """Liste des points à vérifier à la main pour une page."""
    a = []
    if infos["confiance_orientation"] < SEUIL_CONFIANCE_ORIENTATION:
        a.append("orientation incertaine")
    if infos["perspective"] == "echec":
        a.append("perspective non corrigee (coins non trouves)")
    if infos["qualite"]["verdict"] == "a rescanner":
        a.append("qualite insuffisante : " + (", ".join(infos["qualite"]["raisons"]) or "score faible"))
    return a


def pretraiter_page(image, params=None, dpi=None, adapter=True):
    """Prétraite une page. Le défaut principal est diagnostiqué sur l'image d'origine ; si adapter est vrai,
    les réglages de normalisation adaptés à ce défaut sont appliqués (params, s'ils sont fournis, priment).
    Renvoie (gris normalisé, binaire, infos du journal sans fichier ni durée)."""
    gris = en_gris(image)
    defaut, reglages, indices = diagnostiquer(gris)
    effectifs = {**(reglages if adapter else {}), **(params or {})}
    gris, statut, details_persp = corriger_perspective(gris)
    gris, infos = orienter(gris)
    propre, binaire, p = normaliser(gris, effectifs)
    infos.update(diagnostic=defaut, indices_diagnostic=indices, reglages_adaptes=reglages if adapter else {},
                 perspective=statut, details_perspective=details_persp, normalisation=p,
                 qualite=evaluer_qualite(propre, binaire, dpi, infos["confiance_orientation"]),
                 empreinte=doublons.empreinte(binaire),
                 taille_entree=list(image.shape[:2]), taille_sortie=list(binaire.shape[:2]))
    infos["alertes"] = alertes(infos)
    return propre, binaire, infos


def traiter_pdf(chemin, sortie, journal, params=None):
    """Prétraite toutes les pages d'un PDF, écrit les images et le journal. Renvoie les entrées du journal."""
    entrees = []
    for page in pages_pdf(chemin):
        debut = time.perf_counter()
        propre, binaire, infos = pretraiter_page(page["image"], params, dpi=page["dpi_x"])
        ecrire_png(sortie / nom_page(chemin, page["page"]), binaire)
        ecrire_png(sortie / nom_page(chemin, page["page"], "_gris"), propre)
        entree = dict(fichier_source=chemin.name, sha256_source=empreinte(chemin), page=page["page"], **infos,
                      duree_s=round(time.perf_counter() - debut, 3))
        ecrire_entree(journal, entree)
        entrees.append(entree)
    return entrees


def pretraiter_dossier(dossier_donnees, dossier_sortie, params=None):
    """Prétraite tous les PDF du dossier. Renvoie la liste des entrées du journal."""
    verifier_sortie(dossier_donnees, dossier_sortie)
    sortie = Path(dossier_sortie)
    journal = initialiser_journal(sortie / "preprocess_log.jsonl")
    entrees = []
    for chemin in lister_pdf(dossier_donnees):
        entrees += traiter_pdf(chemin, sortie / "preprocessed", journal, params)
    return entrees


def chercher_doublons(dossier_sortie, entrees):
    """Compare les pages prétraitées deux à deux et écrit out/doublons.csv. Renvoie les paires ressemblantes."""
    sortie = Path(dossier_sortie)
    pages = {nom_page(e["fichier_source"], e["page"])[:-4]:
             cv2.imread(str(sortie / "preprocessed" / nom_page(e["fichier_source"], e["page"])), cv2.IMREAD_GRAYSCALE)
             for e in entrees}
    paires = doublons.paires_proches(pages)
    with open(sortie / "doublons.csv", "w", newline="", encoding="utf-8") as fh:
        ecrivain = csv.writer(fh)
        ecrivain.writerow(["page_a", "page_b", "niveau", "distance_empreinte", "ecart_contenu"])
        ecrivain.writerows(paires)
    return paires


def main():
    parser = argparse.ArgumentParser(description="Prétraitement des bons de commande (brique 1).")
    parser.add_argument("donnees", help="dossier des PDF (ex : data/BDC)")
    parser.add_argument("sortie", help="dossier de sortie (ex : out/)")
    args = parser.parse_args()
    entrees = pretraiter_dossier(args.donnees, args.sortie)
    paires = chercher_doublons(args.sortie, entrees)
    print(f"{len(entrees)} pages prétraitées, {sum(bool(e['alertes']) for e in entrees)} avec alerte, "
          f"{sum(p[2] == 'quasi identique' for p in paires)} doublon(s).")


if __name__ == "__main__":
    main()
