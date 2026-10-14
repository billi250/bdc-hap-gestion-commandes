"""Page « À vérifier » : BDC non validés, les plus douteux d'abord, et alertes délai."""
import streamlit as st

from bdc import controles, stockage

from . import commun


def page():
    st.title("À vérifier")
    con = commun.connexion()
    lignes = stockage.lister(con)
    alertes = controles.alertes_delai(lignes, commun.config()["seuils"])
    if alertes:
        st.subheader(f"⏰ Alertes délai ({len(alertes)})")
        for a in alertes:
            col = st.columns([4, 1])
            col[0].warning(f"BDC n° {a['id']} ({a['numero'] or 'sans n°'}, {a['emetteur'] or '?'}) : {a['raison']}")
            with col[1]:
                commun.lien_bdc(a["id"], "Ouvrir")
    file = sorted((l for l in lignes if l["statut"] != "valide"), key=lambda l: l["score"] if l["score"] is not None else 0)
    st.subheader(f"File triée par confiance ({len(file)} BDC non validés)")
    st.caption("Le score baisse avec chaque anomalie : on traite d'abord les BDC les plus douteux.")
    if not file:
        st.success("Rien à vérifier.")
    for l in file:
        col = st.columns([0.8, 4, 1])
        col[0].metric("score", f"{l['score']:.0f}" if l["score"] is not None else "-", label_visibility="collapsed")
        col[1].markdown(commun.resume_bdc(l) + f"  \n<small>{l['fichier']} · reçu le {l['recu_le'][:16].replace('T', ' ')}</small>",
                        unsafe_allow_html=True)
        with col[2]:
            commun.lien_bdc(l["id"], "Revoir")
