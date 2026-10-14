"""Chaîne complète pour un BDC reçu : copie du scan, prétraitement, lecture par le modèle, contrôles,
agent de relecture, enregistrement en base. Utilisée par l'interface et par le benchmark.

Le document déposé est copié dans out/scans/ (jamais dans data/) ; ses pages prétraitées y sont écrites à
côté (gris pour l'affichage et le surlignage, binaire pour la détection visuelle des doublons).
"""
import hashlib
import re
import time
from pathlib import Path

import cv2

from . import agent_relecture, controles, doublons, extraction_llm, stockage
from . import config as cfg
from .fichiers import ecrire_png
from .pdf_io import pages_depuis_octets


def pretraiter(octets, nom):
    """Pages prétraitées : liste de dict(gris, binaire, infos). Même chaîne que la brique 1."""
    from preprocess import pretraiter_page     # point d'entrée de la brique 1, à la racine du projet
    sortie = []
    for page in pages_depuis_octets(octets, nom):
        gris, binaire, infos = pretraiter_page(page["image"], dpi=page["dpi_x"])
        sortie.append(dict(gris=gris, binaire=binaire, infos=dict(page=page["page"], **infos)))
    return sortie


def nom_sur(nom):
    """Nom de fichier sans caractère gênant."""
    return re.sub(r"[^\w.-]+", "_", Path(nom).name)


def archiver(octets, nom, pages, dossier):
    """Copie du document et de ses pages prétraitées. Renvoie (chemin du document, préfixe des pages)."""
    dossier = Path(dossier)
    prefixe = dossier / f"{hashlib.sha256(octets).hexdigest()[:12]}_{Path(nom_sur(nom)).stem}"
    dossier.mkdir(parents=True, exist_ok=True)
    copie = prefixe.with_name(prefixe.name + Path(nom).suffix.lower())
    if not copie.exists():
        copie.write_bytes(octets)
    for p in pages:
        ecrire_png(f"{prefixe}_p{p['infos']['page']}.png", p["gris"])
        ecrire_png(f"{prefixe}_p{p['infos']['page']}_b.png", p["binaire"])
    return copie, prefixe


def pages_archivees(prefixe, binaire=False):
    """Pages prétraitées d'un BDC archivé, dans l'ordre (gris, ou binaire si demandé)."""
    prefixe = Path(prefixe)
    motif = f"{prefixe.name}_p*_b.png" if binaire else f"{prefixe.name}_p*.png"
    fichiers = [f for f in prefixe.parent.glob(motif) if binaire or not f.stem.endswith("_b")]
    fichiers.sort(key=lambda f: int(re.search(r"_p(\d+)", f.stem[len(prefixe.name):]).group(1)))
    return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in fichiers]


def doublons_visuels(binaire_p1, autres):
    """Anomalies « doublon » d'après l'image : page 1 quasi identique à celle d'un BDC déjà en base."""
    a = []
    emp = doublons.empreinte(binaire_p1)
    for o in autres:
        if not o.get("prefixe"):
            continue
        pages = pages_archivees(o["prefixe"], binaire=True)
        if not pages:
            continue
        niveau, _, ecart = doublons.comparer(binaire_p1, pages[0], emp)
        if niveau == "quasi identique":
            a.append(controles.anomalie("doublon", "erreur", f"page 1 quasi identique au BDC n° {o['id']} "
                                                              f"({o['fichier']}) : même document reçu deux fois ?"))
        elif niveau == "a verifier":
            a.append(controles.anomalie("doublon", "attention", f"page 1 très proche du BDC n° {o['id']} "
                                                                f"({o['fichier']}) : à comparer"))
    return a


def traiter(octets, nom, passerelle, con, referentiel, config, rappel=None):
    """Traite un document de bout en bout et l'enregistre. rappel(étape) suit l'avancement (dict avec
    « message »). Renvoie l'id du BDC créé."""
    seuils = config["seuils"]
    noter = rappel or (lambda e: None)
    debut = time.perf_counter()
    noter(dict(type="etape", message="Prétraitement des pages (orientation, redressement, nettoyage)"))
    pages = pretraiter(octets, nom)
    sha = hashlib.sha256(octets).hexdigest()
    copie, prefixe = archiver(octets, nom, pages, cfg.chemin("scans", config))
    reglage = passerelle.reglage("extraction")
    noter(dict(type="etape", message=f"Lecture par {reglage['modele']} ({reglage['fournisseur']}) : "
                                     f"{len(pages)} page(s)"))
    gris = [p["gris"] for p in pages]
    bdc_obj, infos = extraction_llm.extraire_champs(gris, passerelle, referentiel, document=nom)
    extrait = bdc_obj.model_dump()
    bdc = bdc_obj.model_dump()
    autres = [dict(o, prefixe=(str(Path(o["scan"]).with_suffix("")) if o.get("scan") else None))
              for o in stockage.lister(con)]
    visuels = doublons_visuels(pages[0]["binaire"], autres)

    def controle(b):
        return controles.controler(b, referentiel, seuils, sha, autres) + visuels

    noter(dict(type="etape", message="Contrôles automatiques puis agent de relecture"))
    anomalies, journal = agent_relecture.relire(bdc, gris, passerelle, referentiel, seuils, document=nom,
                                                controle=controle, rappel=rappel)
    infos.update(pretraitement=[p["infos"] for p in pages], agent=journal,
                 duree_s=round(time.perf_counter() - debut, 1))
    bdc_id = stockage.enregistrer(con, nom, bdc, extrait, anomalies, controles.score(bdc, anomalies),
                                  controles.statut_initial(anomalies), infos, sha256=sha, scan=str(copie),
                                  modele=f"{infos['fournisseur']}:{infos['modele']}",
                                  relectures=[e for e in journal if e["type"] == "action"])
    noter(dict(type="etape", message=f"Enregistré : BDC n° {bdc_id}"))
    return bdc_id
