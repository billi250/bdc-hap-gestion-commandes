"""Commande unique : inventaire (brique 0), prétraitement (brique 1), planche contact et résumé.

Usage : python run_all.py data/BDC out/
"""
import argparse
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from bdc.fichiers import ecrire_png, nom_page
from bdc.planche import planche_contact
from inventory import inventorier
from preprocess import chercher_doublons, pretraiter_dossier

GRAINE = 0


def fixer_graines(graine=GRAINE):
    """Fixe les graines aléatoires (le traitement est déterministe, c'est une précaution)."""
    random.seed(graine)
    np.random.seed(graine)
    cv2.setRNGSeed(graine)


def faire_planche(dossier_sortie, entrees):
    """Écrit out/contact_sheet.png : page 1 de chaque PDF, avant (out/pages) et après (out/preprocessed)."""
    sortie = Path(dossier_sortie)
    paires = []
    for e in entrees:
        if e["page"] != 1:
            continue
        nom = nom_page(e["fichier_source"], 1)
        avant = cv2.imread(str(sortie / "pages" / nom), cv2.IMREAD_GRAYSCALE)
        apres = cv2.imread(str(sortie / "preprocessed" / nom), cv2.IMREAD_GRAYSCALE)
        titre = (f"{Path(e['fichier_source']).stem[:6]}{Path(e['fichier_source']).stem[-3:]}  {e['diagnostic']}  "
                 f"orient {e['angle_orientation']}  incl {e['angle_redressement']:+.2f}  Q {e['qualite']['score']:.0f}")
        paires.append((avant, apres, titre))
    return ecrire_png(sortie / "contact_sheet.png", planche_contact(paires))


def resumer(entrees, duree, paires=()):
    """Résumé affiché en fin d'exécution."""
    incertaines = [f"{e['fichier_source']} p{e['page']}" for e in entrees if "orientation incertaine" in e["alertes"]]
    persp = [f"{e['fichier_source']} p{e['page']}" for e in entrees if e["perspective"] == "echec"]
    lignes = [f"Pages traitées : {len(entrees)} ({len({e['fichier_source'] for e in entrees})} PDF) en {duree:.1f} s",
              f"Orientation corrigée sans confiance suffisante : {len(incertaines)} {incertaines or ''}",
              f"Perspective non corrigée (photo sans 4 coins) : {len(persp)} {persp or ''}",
              f"Perspective appliquée : {sum(e['perspective'] == 'appliquee' for e in entrees)} page(s)",
              f"Défauts diagnostiqués : {dict(Counter(e['diagnostic'] for e in entrees))}",
              f"Qualité : {dict(Counter(e['qualite']['verdict'] for e in entrees))}",
              f"Doublons (quasi identiques) : {sum(p[2] == 'quasi identique' for p in paires)} ; "
              f"ressemblances fortes à vérifier : {sum(p[2] == 'a verifier' for p in paires)} ; "
              f"même modèle : {sum(p[2] == 'meme modele' for p in paires)} paire(s)"]
    return "\n".join(lignes)


def main():
    parser = argparse.ArgumentParser(description="Briques 0 et 1 : inventaire puis prétraitement.")
    parser.add_argument("donnees", help="dossier des PDF (ex : data/BDC)")
    parser.add_argument("sortie", help="dossier de sortie (ex : out/)")
    args = parser.parse_args()
    fixer_graines()
    debut = time.perf_counter()
    inventorier(args.donnees, args.sortie)
    entrees = pretraiter_dossier(args.donnees, args.sortie)
    paires = chercher_doublons(args.sortie, entrees)
    faire_planche(args.sortie, entrees)
    print(resumer(entrees, time.perf_counter() - debut, paires))


if __name__ == "__main__":
    main()
