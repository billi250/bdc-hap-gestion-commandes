"""Interface Streamlit des laborantins : réception, revue, recherche, assistant et export LIMS des BDC.

Lancer (en local uniquement) :  streamlit run app.py --server.address localhost
La clé Gemini est lue dans le fichier .env (GEMINI_API_KEY=...), jamais dans le code.
"""
import streamlit as st

from bdc import config as cfg
from interface import assistant, commun, export_lims, file, nouveau, pretraitement, recherche, revue, tableau_bord

st.set_page_config(page_title="BDC laboratoire", page_icon="🧪", layout="wide")


def barre_laterale():
    """Nom du laborantin (traçabilité) et choix du modèle : API Gemini ou variante locale RGPD."""
    st.sidebar.text_input("Laborantin", key="laborantin", placeholder="Prénom Nom",
                          help="Écrit dans l'historique de chaque correction et validation.")
    st.sidebar.toggle("Variante locale (aucun envoi externe)", key="locale",
                      help="Lecture par le modèle local (Qwen3-VL via Ollama) : plus lent, moins juste sur le "
                           "manuscrit, mais rien ne quitte ce poste. Pour un client sensible.")
    reglage = commun.passerelle().reglage("extraction")
    if commun.locale():
        st.sidebar.caption(f"Modèle : {reglage['modele']} (local)")
    elif cfg.cle_api():
        st.sidebar.caption(f"Modèle : {reglage['modele']} (Google Gemini, API) · secours : "
                           f"{cfg.modele('secours')['modele']}")
    else:
        st.sidebar.error("Clé Gemini absente : ajouter la ligne GEMINI_API_KEY=... dans le fichier .env du projet, "
                         "puis relancer.")


pages = {
    "Réception": [st.Page(nouveau.page, title="Nouveau BDC", icon="📥", url_path="nouveau", default=True),
                  st.Page(file.page, title="À vérifier", icon="🟠", url_path="a-verifier"),
                  st.Page(revue.page, title="Revue", icon="📄", url_path="revue")],
    "Consultation": [st.Page(recherche.page, title="Recherche", icon="🔍", url_path="recherche"),
                     st.Page(assistant.page, title="Assistant", icon="💬", url_path="assistant"),
                     st.Page(tableau_bord.page, title="Tableau de bord", icon="📊", url_path="tableau-de-bord")],
    "Sorties": [st.Page(export_lims.page, title="Export LIMS", icon="📤", url_path="export-lims"),
                st.Page(pretraitement.page, title="Prétraitement (local)", icon="🧹", url_path="pretraitement")],
}
commun.PAGES.update({p.url_path: p for groupe in pages.values() for p in groupe})
navigation = st.navigation(pages)
barre_laterale()
navigation.run()
