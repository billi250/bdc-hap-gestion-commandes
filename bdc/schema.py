"""Schéma du bon de commande, imposé au modèle et vérifié en Python (pydantic).

Le même schéma sert trois fois : il est envoyé au modèle comme format de réponse obligatoire, il valide la
réponse (si une nouvelle version du modèle répond autrement, l'erreur est détectée tout de suite), et il
structure ce qui est stocké en base puis exporté vers le LIMS.

Boîtes : [ymin, xmin, ymax, xmax] normalisées de 0 à 1000 sur la page (convention de Gemini), pour surligner
sur le scan la zone d'où vient chaque valeur.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DELAIS = ("standard", "normal", "urgent", "express")


def normaliser_boite(v):
    """Boîte ramenée dans [0, 1000] ; None si elle n'a pas 4 valeurs ou si elle est vide."""
    if not v or len(v) != 4:
        return None
    v = [min(1000, max(0, int(x))) for x in v]
    return v if v[0] < v[2] and v[1] < v[3] else None


class Champ(BaseModel):
    """Une valeur lue sur le scan, avec sa confiance et sa position."""
    valeur: str | None = Field(None, description="Texte recopié tel qu'écrit ; null si absent ou illisible.")
    confiance: int = Field(0, ge=0, le=100, description="Certitude de la lecture, de 0 à 100.")
    page: int = Field(1, ge=1, description="Numéro de page (1 pour la première).")
    boite: list[int] | None = Field(None, description="[ymin, xmin, ymax, xmax] de 0 à 1000 sur la page.")

    @field_validator("boite")
    @classmethod
    def _boite(cls, v):
        return normaliser_boite(v)

    @field_validator("valeur")
    @classmethod
    def _valeur(cls, v):
        v = (v or "").strip()
        return v or None


class Echantillon(BaseModel):
    """Une ligne du tableau des échantillons."""
    reference: str | None = Field(None, description="Référence de l'échantillon telle qu'écrite.")
    profondeur: str | None = Field(None, description="Profondeur en cm, ex. « 0-30 ».")
    analyses: list[str] = Field(default_factory=list, description="Codes des analyses demandées (cases cochées).")
    observations: str | None = None
    ligne_barree: bool = Field(False, description="Vrai si la ligne est rayée d'un trait (annulée).")
    confiance: int = Field(0, ge=0, le=100)
    page: int = Field(1, ge=1)
    boite: list[int] | None = Field(None, description="[ymin, xmin, ymax, xmax] de la ligne, de 0 à 1000.")

    @field_validator("boite")
    @classmethod
    def _boite(cls, v):
        return normaliser_boite(v)


CHAMPS = {  # nom -> (groupe affiché, libellé)
    "numero_commande": ("Commande", "N° de commande"),
    "date_commande": ("Commande", "Date de commande"),
    "date_prelevement": ("Commande", "Date de prélèvement"),
    "delai": ("Commande", "Délai"),
    "nb_echantillons_annonce": ("Commande", "Nombre d'échantillons annoncé"),
    "emetteur_nom": ("Émetteur (donneur d'ordre)", "Nom"),
    "emetteur_adresse": ("Émetteur (donneur d'ordre)", "Adresse"),
    "emetteur_siret": ("Émetteur (donneur d'ordre)", "SIRET"),
    "emetteur_telephone": ("Émetteur (donneur d'ordre)", "Téléphone"),
    "emetteur_email": ("Émetteur (donneur d'ordre)", "E-mail"),
    "charge_affaires": ("Chargé d'affaires", "Nom"),
    "client_nom": ("Client final (site)", "Nom"),
    "client_adresse_site": ("Client final (site)", "Adresse du site"),
    "client_parcelles": ("Client final (site)", "Parcelles"),
    "client_commune": ("Client final (site)", "Commune"),
    "client_telephone": ("Client final (site)", "Téléphone"),
    "client_interlocuteur": ("Client final (site)", "Interlocuteur"),
    "laboratoire": ("Laboratoire", "Destinataire"),
}
OBLIGATOIRES = ("numero_commande", "date_commande", "emetteur_nom")


class BonDeCommande(BaseModel):
    """Tout ce qu'on extrait d'un bon de commande."""
    numero_commande: Champ = Field(default_factory=Champ)
    date_commande: Champ = Field(default_factory=Champ, description="JJ/MM/AAAA")
    date_prelevement: Champ = Field(default_factory=Champ, description="JJ/MM/AAAA")
    delai: Champ = Field(default_factory=Champ, description="standard, normal, urgent ou express")
    nb_echantillons_annonce: Champ = Field(default_factory=Champ,
                                           description="Nombre d'échantillons écrit sur le bon (pas compté).")
    emetteur_nom: Champ = Field(default_factory=Champ, description="Organisme de l'en-tête, qui passe commande.")
    emetteur_adresse: Champ = Field(default_factory=Champ)
    emetteur_siret: Champ = Field(default_factory=Champ, description="14 chiffres.")
    emetteur_telephone: Champ = Field(default_factory=Champ, description="0X XX XX XX XX")
    emetteur_email: Champ = Field(default_factory=Champ)
    charge_affaires: Champ = Field(default_factory=Champ, description="Personne indiquée « chargé d'affaires ».")
    client_nom: Champ = Field(default_factory=Champ, description="Client final ou site pour qui sont les analyses.")
    client_adresse_site: Champ = Field(default_factory=Champ)
    client_parcelles: Champ = Field(default_factory=Champ)
    client_commune: Champ = Field(default_factory=Champ)
    client_telephone: Champ = Field(default_factory=Champ, description="0X XX XX XX XX")
    client_interlocuteur: Champ = Field(default_factory=Champ)
    laboratoire: Champ = Field(default_factory=Champ, description="Laboratoire destinataire.")
    echantillons: list[Echantillon] = Field(default_factory=list,
                                            description="Une entrée par ligne remplie du tableau, dans l'ordre.")

    @field_validator("delai")
    @classmethod
    def _delai(cls, c):
        t = (c.valeur or "").lower()
        trouve = next((d for d in DELAIS if d in t), None)
        return c.model_copy(update={"valeur": trouve or c.valeur})


class Relecture(BaseModel):
    """Réponse de l'agent de relecture pour un champ zoomé."""
    valeur: str | None = Field(None, description="Valeur relue sur le zoom ; null si illisible.")
    confiance: int = Field(0, ge=0, le=100)
    explication: str = Field("", description="Une phrase : ce qui a été lu et pourquoi.")


Statut = Literal["brouillon", "a verifier", "valide"]
