"""Page « Export LIMS » : CSV et JSON des seuls BDC validés."""
from datetime import datetime

import pandas as pd
import streamlit as st

from bdc import export, stockage

from . import commun


def page():
    st.title("Export vers le LIMS")
    con = commun.connexion()
    valides = [stockage.lire(con, l["id"]) for l in stockage.lister(con, "valide")]
    attente = len(stockage.lister(con)) - len(valides)
    st.caption(f"Seuls les BDC validés partent dans le LIMS : {len(valides)} validé(s), {attente} en attente. "
               "Une ligne par analyse à réaliser ; les échantillons annulés sont exclus.")
    if not valides:
        st.info("Aucun BDC validé pour l'instant.")
        return
    ref = commun.referentiel()
    lignes = export.lignes_lims(valides, ref)
    st.dataframe(pd.DataFrame(lignes, columns=export.COLONNES_LIMS), hide_index=True, width="stretch")
    horodatage = f"{datetime.now():%Y%m%d_%H%M}"
    c = st.columns(2)
    c[0].download_button("Télécharger le CSV LIMS", export.csv_lims(valides, ref), f"lims_{horodatage}.csv",
                         mime="text/csv", type="primary", width="stretch")
    c[1].download_button("Télécharger le JSON", export.json_bdc(valides), f"bdc_valides_{horodatage}.json",
                         mime="application/json", width="stretch")
