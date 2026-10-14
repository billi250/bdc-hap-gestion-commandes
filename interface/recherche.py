"""Page « Recherche » : par n° de BDC, client, date, échantillon ou analyse."""
import pandas as pd
import streamlit as st

from bdc import stockage

from . import commun


def page():
    st.title("Recherche")
    con = commun.connexion()
    texte = st.text_input("Recherche libre", placeholder="n° de commande, client, référence d'échantillon, analyse...")
    with st.expander("Critères précis", expanded=False):
        c = st.columns(3)
        numero = c[0].text_input("N° de commande")
        client = c[1].text_input("Client (émetteur ou client final)")
        statut = c[2].selectbox("Statut", ["", "brouillon", "a verifier", "valide"],
                                format_func=lambda s: commun.STATUTS.get(s, "tous"))
        c = st.columns(4)
        date_min = c[0].text_input("Date de commande après le (JJ/MM/AAAA)")
        date_max = c[1].text_input("avant le (JJ/MM/AAAA)")
        echantillon = c[2].text_input("Référence d'échantillon")
        analyse = c[3].selectbox("Analyse", [""] + [a["code"] for a in commun.referentiel()["analyses"]])
    trouves = stockage.chercher(con, texte=texte or None, numero=numero or None, client=client or None,
                                statut=statut or None, date_min=date_min or None, date_max=date_max or None,
                                echantillon=echantillon or None, analyse=analyse or None)
    st.caption(f"{len(trouves)} BDC trouvé(s)")
    if trouves:
        tableau = pd.DataFrame(trouves)[["id", "statut", "numero", "emetteur", "client", "date_commande", "delai",
                                         "nb_echantillons", "score", "fichier"]]
        tableau.insert(0, "lien", [f"/revue?bdc={i}" for i in tableau["id"]])
        st.dataframe(tableau, hide_index=True, width="stretch",
                     column_config={"lien": st.column_config.LinkColumn("", display_text="Ouvrir")})
