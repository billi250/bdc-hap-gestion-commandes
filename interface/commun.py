"""Éléments partagés par les pages de l'interface : base, passerelle, référentiel, dessin des zones."""
from pathlib import Path

import cv2
import streamlit as st

from bdc import config as cfg
from bdc import stockage
from bdc.modeles import Passerelle

PAGES = {}       # nom -> st.Page, rempli par app.py (liens entre pages)
STATUTS = {"brouillon": "Brouillon", "a verifier": "À vérifier", "valide": "Validé"}
PASTILLES = {"brouillon": "⚪", "a verifier": "🟠", "valide": "🟢"}
ORANGE, BLEU, VERT = (0, 140, 255), (255, 90, 0), (60, 170, 60)    # BGR


@st.cache_resource
def connexion():
    """Connexion SQLite partagée (une par serveur Streamlit)."""
    return stockage.ouvrir(cfg.chemin("base"))


def config():
    return cfg.charger()


def referentiel():
    return cfg.referentiel()


def locale():
    """Vrai si la variante locale (aucun envoi externe) est choisie dans la barre latérale."""
    return st.session_state.get("locale", False)


def passerelle():
    """Passerelle vers le modèle choisi (Gemini, ou local en variante RGPD)."""
    return Passerelle(config(), locale=locale())


def laborantin():
    """Nom saisi dans la barre latérale (tracé dans l'historique des corrections)."""
    return (st.session_state.get("laborantin") or "").strip()


def lien_bdc(bdc_id, libelle=None, cle=None):
    """Lien vers l'écran de revue d'un BDC (le scan et ses champs)."""
    st.page_link(PAGES["revue"], label=libelle or f"Ouvrir le BDC n° {bdc_id}", icon="📄",
                 query_params={"bdc": str(bdc_id)})


def couleur_boite(etat):
    return {"choisi": BLEU, "anomalie": ORANGE}.get(etat, VERT)


def dessiner(page, boites):
    """Page (niveaux de gris) avec des rectangles : boites = [(boîte 0-1000, état, étiquette)]."""
    image = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    h, w = page.shape[:2]
    epais = max(2, w // 400)
    for boite, etat, etiquette in sorted(boites, key=lambda b: b[1] == "choisi"):
        y0, x0, y1, x1 = boite
        p0, p1 = (int(x0 * w / 1000), int(y0 * h / 1000)), (int(x1 * w / 1000), int(y1 * h / 1000))
        couleur = couleur_boite(etat)
        cv2.rectangle(image, p0, p1, couleur, epais * (2 if etat == "choisi" else 1))
        if etiquette and etat != "normal":
            cv2.putText(image, etiquette, (p0[0], max(12, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * epais / 2,
                        couleur, max(1, epais // 2), cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prefixe_pages(d):
    """Préfixe des pages prétraitées archivées d'un BDC (out/scans/<sha>_<nom>)."""
    return Path(d["scan"]).with_suffix("") if d.get("scan") else None


def resume_bdc(l):
    """Libellé court d'un BDC pour les listes."""
    return (f"{PASTILLES.get(l['statut'], '')} n° {l['id']} : {l['numero'] or 'sans n°'} · {l['emetteur'] or '?'} · "
            f"{l['date_commande'] or '?'} · {STATUTS.get(l['statut'], l['statut'])}")
