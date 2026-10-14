"""Lecture de config.yaml (modèles, seuils, chemins), du référentiel et de la clé d'API.

La clé n'est jamais écrite dans le code ni dans la configuration : elle est lue dans la variable
d'environnement GEMINI_API_KEY (ou GOOGLE_API_KEY), ou dans un fichier .env non versionné.
"""
import os
from functools import lru_cache
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parent.parent
VARIABLES_CLE = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


@lru_cache(maxsize=4)
def charger(chemin=None):
    """Configuration complète (dict). chemin : config.yaml par défaut, à la racine du projet."""
    with open(chemin or RACINE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def modele(usage, config=None):
    """Fournisseur, identifiant figé et niveau de réflexion pour un usage (extraction, chatbot...)."""
    return dict((config or charger())[usage])


def seuil(nom, config=None):
    """Valeur d'un seuil de config.yaml."""
    return (config or charger())["seuils"][nom]


def chemin(nom, config=None):
    """Chemin de config.yaml, relatif à la racine du projet."""
    return RACINE / (config or charger())["chemins"][nom]


@lru_cache(maxsize=2)
def referentiel(chemin_fichier=None):
    """Référentiel (laboratoires, clients, analyses). Repli sur l'exemple versionné si le vrai est absent."""
    reel = Path(chemin_fichier) if chemin_fichier else chemin("referentiel")
    fichier = reel if reel.exists() else RACINE / "referentiel.exemple.yaml"
    with open(fichier, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _lire_env(fichier):
    """Variables d'un fichier .env (lignes NOM=valeur), sans les exporter."""
    valeurs = {}
    if fichier.exists():
        for ligne in fichier.read_text(encoding="utf-8").splitlines():
            if "=" in ligne and not ligne.lstrip().startswith("#"):
                nom, valeur = ligne.split("=", 1)
                valeurs[nom.strip()] = valeur.strip().strip("'\"")
    return valeurs


def cle_api():
    """Clé Gemini : variable d'environnement, puis .env du projet, puis ~/.env. None si absente."""
    for source in (os.environ, _lire_env(RACINE / ".env"), _lire_env(Path.home() / ".env")):
        for nom in VARIABLES_CLE:
            if source.get(nom):
                return source[nom]
    return None
