"""Page « Tableau de bord » : BDC reçus, validés, en attente, délai moyen par client ; envois externes."""
import pandas as pd
import streamlit as st

from bdc import stockage
from bdc.modeles import lire_envois

from . import commun


def page():
    st.title("Tableau de bord")
    con = commun.connexion()
    s = stockage.statistiques(con)
    c = st.columns(4)
    c[0].metric("BDC reçus", s["recus"])
    c[1].metric("Validés", s["par_statut"]["valide"])
    c[2].metric("À vérifier", s["par_statut"]["a verifier"])
    c[3].metric("Brouillons", s["par_statut"]["brouillon"])
    if s["par_emetteur"]:
        st.subheader("Par client")
        tableau = pd.DataFrame([dict(client=k, **v) for k, v in s["par_emetteur"].items()]).sort_values("recus", ascending=False)
        st.dataframe(tableau, hide_index=True, width="stretch",
                     column_config={"delai_moyen_validation_h": st.column_config.NumberColumn(
                         "délai moyen de validation (h)", format="%.1f")})
        st.bar_chart(tableau.set_index("client")[["valides", "en_attente"]], stack=True)
    st.subheader("Envois à un service externe")
    st.caption("Chaque image envoyée au modèle par API est tracée avant l'envoi : date, document, modèle, "
               "empreinte SHA-256 des images (pas les images elles-mêmes) et justification.")
    envois = lire_envois()
    if envois:
        df = pd.DataFrame(envois)
        envoyes = df[df["statut"] == "envoi"]
        c = st.columns(3)
        c[0].metric("Envois", len(envoyes))
        c[1].metric("Images envoyées", int(envoyes["images_sha256"].map(len).sum()))
        c[2].metric("Documents concernés", envoyes["document"].nunique())
        st.dataframe(df.iloc[::-1].drop(columns=["images_sha256"]), hide_index=True, width="stretch")
    else:
        st.caption("Aucun envoi externe.")
