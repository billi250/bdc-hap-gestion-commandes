"""Page « Nouveau BDC » : dépôt d'un document, lecture par le modèle, contrôles et agent de relecture."""
from pathlib import Path

import streamlit as st

from bdc.modeles import ModeleIndisponible
from bdc.pdf_io import EXTENSIONS_IMAGE, lister_pdf
from bdc.traitement import traiter

from . import commun

FORMATS = ["pdf"] + sorted(e.lstrip(".") for e in EXTENSIONS_IMAGE)
DOSSIER_EXEMPLES = Path("data/BDC")


def documents():
    """Documents déposés, ou un exemple du corpus local. Renvoie [(nom, octets)]."""
    fichiers = st.file_uploader("Déposez un ou plusieurs bons de commande (PDF ou image)", type=FORMATS,
                                accept_multiple_files=True)
    if fichiers:
        return [(f.name, f.getvalue()) for f in fichiers]
    exemples = lister_pdf(DOSSIER_EXEMPLES) if DOSSIER_EXEMPLES.is_dir() else []
    if exemples:
        choix = st.selectbox("Ou choisissez un exemple du corpus local", ["(aucun)"] + [p.name for p in exemples])
        if choix != "(aucun)":
            return [(choix, (DOSSIER_EXEMPLES / choix).read_bytes())]
    return []


def avertissement_envoi():
    """Ce qui part à l'extérieur, dit avant l'envoi."""
    reglage = commun.passerelle().reglage("extraction")
    if commun.locale():
        st.info(f"Variante locale : lecture par {reglage['modele']} sur ce poste. Aucun envoi externe "
                "(plus lent, environ 1 minute par page).")
    else:
        st.warning(f"Les pages prétraitées seront envoyées à Google ({reglage['modele']}) pour être lues. "
                   "Chaque envoi est tracé (date, document, empreinte, justification) : voir « Envois externes ».")


def page():
    st.title("Nouveau bon de commande")
    st.caption("Prétraitement, lecture du scan par le modèle (imprimé et manuscrit), contrôles automatiques, "
               "puis relecture par l'agent des champs douteux. Le BDC part ensuite dans la file « à vérifier ».")
    docs = documents()
    if not docs:
        st.info("Déposez un document pour commencer.")
        return
    avertissement_envoi()
    if not st.button(f"Lire et contrôler {len(docs)} document(s)", type="primary"):
        return
    con, ref, conf, passerelle = commun.connexion(), commun.referentiel(), commun.config(), commun.passerelle()
    for nom, octets in docs:
        with st.status(f"{nom}", expanded=True) as statut:
            def rappel(etape):
                icone = {"plan": "🧭", "action": "🔎", "fin": "✋", "bilan": "📋"}.get(etape.get("type"), "⏳")
                st.write(f"{icone} {etape['message']}")
            try:
                bdc_id = traiter(octets, nom, passerelle, con, ref, conf, rappel)
            except (ModeleIndisponible, ValueError) as e:
                statut.update(label=f"{nom} : échec", state="error")
                st.error(str(e))
                continue
            statut.update(label=f"{nom} : BDC n° {bdc_id} enregistré", state="complete", expanded=False)
        commun.lien_bdc(bdc_id, f"Revoir le BDC n° {bdc_id} ({nom})")
