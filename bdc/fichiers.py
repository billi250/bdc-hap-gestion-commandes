"""Petits utilitaires de fichiers : empreinte, écriture d'images, garde-fou sur le dossier des scans."""
import hashlib
from pathlib import Path

import cv2


def empreinte(chemin):
    """Empreinte SHA-256 d'un fichier (sert à prouver que les scans d'origine ne sont pas modifiés)."""
    h = hashlib.sha256()
    with open(chemin, "rb") as fh:
        for bloc in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def verifier_sortie(dossier_donnees, dossier_sortie):
    """Refuse d'écrire dans le dossier des scans d'origine ou dans un de ses sous-dossiers."""
    donnees, sortie = Path(dossier_donnees).resolve(), Path(dossier_sortie).resolve()
    if sortie == donnees or donnees in sortie.parents:
        raise ValueError(f"Le dossier de sortie {sortie} est dans le dossier des scans {donnees} : refusé.")


def ecrire_png(chemin, image):
    """Écrit une image PNG en créant les dossiers parents si besoin."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(chemin), image):
        raise IOError(f"Écriture impossible : {chemin}")
    return chemin


def nom_page(chemin_pdf, page, suffixe=""):
    """Nom de fichier d'une page : <pdf sans extension>_p<n><suffixe>.png."""
    return f"{Path(chemin_pdf).stem}_p{page}{suffixe}.png"
