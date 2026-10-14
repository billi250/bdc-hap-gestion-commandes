"""Page « Assistant » : questions en français courant sur les BDC, réponses avec le n° de BDC cité."""
from datetime import date

import streamlit as st

from bdc.agent_chat import outils_par_defaut, repondre
from bdc.modeles import ModeleIndisponible

from . import commun

EXEMPLES = ["Quels BDC sont en attente de validation ?", "Quels BDC urgents ne sont pas encore validés ?",
            "Combien de BDC avons-nous reçus par client ?", "Quelles analyses demande la commande 004587 ?"]


def afficher(message):
    with st.chat_message(message["role"]):
        st.markdown(message["texte"])
        if message.get("appels"):
            with st.expander(f"🔧 {len(message['appels'])} appel(s) à la base"):
                for nom, args, resultat in message["appels"]:
                    st.markdown(f"**{nom}**({', '.join(f'{k}={v!r}' for k, v in args.items())})")
                    st.json(resultat, expanded=False)
        for i in message.get("sources", [])[:8]:
            commun.lien_bdc(i, f"Source : BDC n° {i}")


def page():
    st.title("Assistant")
    st.caption("Il interroge la base (il ne devine pas), cite le n° de BDC et ne modifie rien. "
               "Les corrections passent par l'écran de revue.")
    if commun.locale():
        st.info("Indisponible en variante locale : l'assistant utilise le modèle par API (appel de fonctions).")
        return
    st.session_state.setdefault("conversation", [])
    for m in st.session_state.conversation:
        afficher(m)
    question = st.chat_input("Votre question sur les BDC")
    if not st.session_state.conversation and not question:
        choix = st.pills("Exemples", EXEMPLES)
        question = choix
    if not question:
        return
    message = dict(role="user", texte=question)
    st.session_state.conversation.append(message)
    afficher(message)
    historique = [("user" if m["role"] == "user" else "model", m["texte"]) for m in st.session_state.conversation[:-1]][-10:]
    with st.spinner("L'agent consulte la base..."):
        try:
            r = repondre(question, commun.passerelle(), outils_par_defaut(commun.connexion()), historique,
                         date.today().strftime("%d/%m/%Y"))
        except ModeleIndisponible as e:
            r = dict(reponse=f"Modèle indisponible : {e}", appels=[], sources=[])
    reponse = dict(role="assistant", texte=r["reponse"] or "(pas de réponse)", appels=r["appels"], sources=r["sources"])
    st.session_state.conversation.append(reponse)
    afficher(reponse)
    if st.button("Nouvelle conversation"):
        st.session_state.conversation = []
        st.rerun()
