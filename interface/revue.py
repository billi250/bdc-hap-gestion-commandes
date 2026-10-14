"""Page « Revue » : le scan à gauche, les champs extraits à droite, modifiables ; validation du BDC.

- Les champs douteux (règle échouée, confiance basse) sont marqués 🟠 et listés en premier.
- Cocher « voir » sur un champ ou un échantillon surligne en bleu sa zone sur le scan.
- « + » sous le tableau des échantillons ajoute un échantillon ; les analyses se choisissent dans le
  référentiel. Le client se choisit aussi dans le référentiel (auto-complétion).
- Chaque modification est tracée (valeur extraite, valeur corrigée, qui, quand). Seul un BDC validé part
  dans le CSV LIMS ; sa version validée enrichit le jeu de référence du benchmark.
"""
import copy
from pathlib import Path

import pandas as pd
import streamlit as st

from bdc import config as cfg
from bdc import controles, gcms, stockage, verite
from bdc.pdf_io import pages_depuis_octets
from bdc.schema import CHAMPS
from bdc.traitement import pages_archivees

from . import commun


def choisir_bdc(con):
    """BDC à revoir : celui du lien (?bdc=), sinon choisi dans la liste."""
    lignes = stockage.lister(con)
    if not lignes:
        st.info("Aucun BDC en base : commencez par « Nouveau BDC ».")
        return None
    ids = [l["id"] for l in lignes]
    demande = st.query_params.get("bdc")
    defaut = ids.index(int(demande)) if demande and demande.isdigit() and int(demande) in ids else 0
    choix = st.selectbox("BDC", lignes, index=defaut, format_func=commun.resume_bdc)
    st.query_params["bdc"] = str(choix["id"])
    return choix["id"]


def etat(nom, champ, en_anomalie, seuil):
    if nom in en_anomalie:
        return "🟠"
    if champ.get("valeur") and champ.get("confiance", 100) < seuil:
        return "🟠"
    return "🟢" if champ.get("valeur") else "⚪"


def table_champs(bdc, en_anomalie, seuil):
    lignes = [dict(voir=False, etat=etat(n, bdc.get(n) or {}, en_anomalie, seuil), champ=f"{g.split(' (')[0]} · {l}",
                   valeur=(bdc.get(n) or {}).get("valeur") or "", confiance=(bdc.get(n) or {}).get("confiance"),
                   cle=n) for n, (g, l) in CHAMPS.items()]
    return pd.DataFrame(sorted(lignes, key=lambda r: r["etat"] != "🟠"))


def table_echantillons(bdc, en_anomalie, seuil):
    lignes = []
    for i, e in enumerate(bdc.get("echantillons") or [], 1):
        douteux = f"echantillons[{i}]" in en_anomalie or e.get("confiance", 100) < seuil
        lignes.append(dict(voir=False, etat="🟠" if douteux else "🟢", reference=e.get("reference") or "",
                           profondeur=e.get("profondeur") or "", analyses=e.get("analyses") or [],
                           observations=e.get("observations") or "", annule=controles.annule(e),
                           confiance=e.get("confiance")))
    return pd.DataFrame(lignes, columns=["voir", "etat", "reference", "profondeur", "analyses", "observations",
                                         "annule", "confiance"])


def pages_affichees(d, originales):
    """Pages prétraitées (zones exactes) ou pages du scan d'origine (lecture seule)."""
    if originales and d.get("scan") and Path(d["scan"]).exists():
        return [p["image"] for p in pages_depuis_octets(Path(d["scan"]).read_bytes(), d["fichier"])]
    return pages_archivees(commun.prefixe_pages(d)) if d.get("scan") else []


def zones(bdc, champs_vus, ech_vus, en_anomalie, page_no, toutes):
    """Boîtes à dessiner sur la page : choisies (bleu), en anomalie (orange), les autres si demandé."""
    sortie = []
    for n, (_, libelle) in CHAMPS.items():
        c = bdc.get(n) or {}
        if not c.get("boite") or c.get("page", 1) != page_no:
            continue
        e = "choisi" if n in champs_vus else "anomalie" if n in en_anomalie else "normal"
        if e != "normal" or toutes:
            sortie.append((c["boite"], e, libelle))
    for i, ech in enumerate(bdc.get("echantillons") or [], 1):
        if not ech.get("boite") or ech.get("page", 1) != page_no:
            continue
        e = "choisi" if i - 1 in ech_vus else "anomalie" if f"echantillons[{i}]" in en_anomalie else "normal"
        if e != "normal" or toutes:
            sortie.append((ech["boite"], e, ech.get("reference") or f"échantillon {i}"))
    return sortie


def appliquer(bdc, champs_edites, ech_edites, client_choisi):
    """Nouvelle version du BDC d'après les tableaux modifiés (valeur corrigée : confiance 100)."""
    nouveau = copy.deepcopy(bdc)
    for r in champs_edites.to_dict("records"):
        n, valeur = r["cle"], (r["valeur"] or "").strip() or None
        ancien = nouveau.get(n) or {}
        if valeur != ancien.get("valeur"):
            nouveau[n] = {**ancien, "valeur": valeur, "confiance": 100}
    if client_choisi and client_choisi != (nouveau["emetteur_nom"] or {}).get("valeur"):
        nouveau["emetteur_nom"] = {**nouveau["emetteur_nom"], "valeur": client_choisi, "confiance": 100}
    anciens = bdc.get("echantillons") or []
    echantillons = []
    for idx, r in ech_edites.iterrows():
        base = dict(anciens[idx]) if isinstance(idx, int) and idx < len(anciens) else dict(confiance=100, page=1, boite=None)
        modifie = ((r["reference"] or None) != base.get("reference") or (r["profondeur"] or None) != base.get("profondeur")
                   or list(r["analyses"] or []) != (base.get("analyses") or []))
        base.update(reference=(r["reference"] or "").strip() or None, profondeur=(r["profondeur"] or "").strip() or None,
                    analyses=list(r["analyses"] or []), observations=(r["observations"] or "").strip() or None,
                    ligne_barree=bool(r["annule"]))
        if modifie:
            base["confiance"] = 100
        if base["reference"] or base["analyses"]:
            echantillons.append(base)
    nouveau["echantillons"] = echantillons
    return nouveau


def recontroler(con, d, nouveau):
    """Anomalies de la version corrigée (les doublons visuels détectés à la réception sont conservés)."""
    autres = [l for l in stockage.lister(con) if l["id"] != d["id"]]
    visuels = [a for a in d["anomalies"] or [] if a["champ"] == "doublon" and a["message"].startswith("page 1")]
    anomalies = controles.controler(nouveau, commun.referentiel(), commun.config()["seuils"], d["sha256"], autres) + visuels
    return anomalies, controles.score(nouveau, anomalies)


def afficher_anomalies(anomalies):
    erreurs = [a for a in anomalies if a["niveau"] == "erreur"]
    attentions = [a for a in anomalies if a["niveau"] != "erreur"]
    if not anomalies:
        st.success("Aucune anomalie : le BDC peut être validé.")
    for a in erreurs:
        st.error(a["message"], icon="⛔")
    if attentions:
        with st.expander(f"🟠 {len(attentions)} point(s) d'attention", expanded=not erreurs):
            for a in attentions:
                st.markdown(f"- {a['message']}")


def page():
    st.title("Revue d'un BDC")
    con = commun.connexion()
    bdc_id = choisir_bdc(con)
    if bdc_id is None:
        return
    d = stockage.lire(con, bdc_id)
    bdc, seuil = d["courant"], commun.config()["seuils"]["confiance_min"]
    en_anomalie = {a["champ"] for a in d["anomalies"] or []}
    c = st.columns(5)
    c[0].metric("Statut", commun.STATUTS.get(d["statut"], d["statut"]))
    c[1].metric("Score de confiance", f"{d['score']:.0f} / 100" if d["score"] is not None else "-")
    c[2].metric("Échantillons à analyser", d["nb_echantillons"])
    c[3].metric("Reçu le", d["recu_le"][:16].replace("T", " "))
    c[4].metric("Lu par", (d["modele"] or "-").split(":")[-1])
    if d["statut"] == "valide":
        st.success(f"Validé par {d['valide_par']} le {d['valide_le'][:16].replace('T', ' ')}")
    afficher_anomalies(d["anomalies"] or [])

    gauche, droite = st.columns([1, 1.15])
    with droite:
        onglets = st.tabs(["Champs", "Échantillons", "Historique", "Agent", "Lecture brute"])
        with onglets[0]:
            st.caption("🟠 à vérifier en priorité · cochez « voir » pour surligner la zone sur le scan.")
            champs = st.data_editor(table_champs(bdc, en_anomalie, seuil), hide_index=True, width="stretch",
                                    key=f"champs_{bdc_id}", disabled=["etat", "champ", "confiance"],
                                    column_config={"cle": None, "voir": st.column_config.CheckboxColumn("voir", width="small"),
                                                   "champ": st.column_config.TextColumn("champ", width="medium"),
                                                   "valeur": st.column_config.TextColumn("valeur", width="large"),
                                                   "etat": st.column_config.TextColumn("", width="small"),
                                                   "confiance": st.column_config.ProgressColumn("confiance", min_value=0,
                                                                                                max_value=100, format="%d")})
            clients = [c["nom"] for c in commun.referentiel().get("clients", [])]
            actuel = (bdc.get("emetteur_nom") or {}).get("valeur")
            client = st.selectbox("Client (émetteur) choisi dans le référentiel", [""] + clients,
                                  index=(clients.index(actuel) + 1) if actuel in clients else 0,
                                  help="Tapez quelques lettres pour filtrer. Laisser vide : garder la valeur du tableau.")
        with onglets[1]:
            st.caption("« + » en bas du tableau : ajouter un échantillon. Analyses tirées du référentiel. "
                       "« annule » : ligne barrée sur le bon (à ne pas analyser).")
            codes = [a["code"] for a in commun.referentiel()["analyses"]]
            ech = st.data_editor(table_echantillons(bdc, en_anomalie, seuil), hide_index=True, width="stretch",
                                 num_rows="dynamic", key=f"ech_{bdc_id}", disabled=["etat", "confiance"],
                                 column_config={"voir": st.column_config.CheckboxColumn("voir", width="small"),
                                                "etat": st.column_config.TextColumn("", width="small"),
                                                "analyses": st.column_config.MultiselectColumn("analyses", options=codes),
                                                "annule": st.column_config.CheckboxColumn("annulé"),
                                                "confiance": st.column_config.ProgressColumn("confiance", min_value=0,
                                                                                             max_value=100, format="%d")})
            references = [r for r, a in zip(ech["reference"], ech["annule"]) if r and not a]
            with st.expander("Résultats HAP (GC-MS) de ces échantillons"):
                for ref in references:
                    res = gcms.resultats(ref, cfg.chemin("gcms_resultats"))
                    if not res["disponible"]:
                        st.caption(res["message"])
                        break
                    st.markdown(f"**{ref}**")
                    st.dataframe(pd.DataFrame(res["resultats"]), hide_index=True) if res["resultats"] else st.caption(res["message"])
        with onglets[2]:
            hist = stockage.historique(con, bdc_id)
            st.dataframe(pd.DataFrame(hist), hide_index=True, width="stretch") if hist else st.caption("Aucune correction.")
        with onglets[3]:
            for e in (d["infos"] or {}).get("agent", []):
                st.markdown(f"{ {'plan': '🧭', 'action': '🔎', 'fin': '✋', 'bilan': '📋'}.get(e['type'], '·') } {e['message']}")
        with onglets[4]:
            st.caption("Lecture du modèle avant tout recalage (figée), puis recalages sur le référentiel.")
            st.json((d["infos"] or {}).get("recalages", []), expanded=False)
            st.json(d["extrait"], expanded=False)

    with gauche:
        originales = st.toggle("Scan d'origine (sans les zones)", value=False)
        pages = pages_affichees(d, originales)
        if not pages:
            st.warning("Scan introuvable.")
        else:
            champs_vus = set(champs.loc[champs["voir"], "cle"])
            ech_vus = {i for i, v in enumerate(ech["voir"]) if v}
            pages_vues = [(bdc.get(n) or {}).get("page", 1) for n in champs_vus] + \
                         [(bdc.get("echantillons") or [{}])[i].get("page", 1) for i in ech_vus if i < len(bdc.get("echantillons") or [])]
            no = st.radio("Page", list(range(1, len(pages) + 1)), horizontal=True,
                          index=min(len(pages), pages_vues[0] if pages_vues else 1) - 1) if len(pages) > 1 else 1
            image = pages[no - 1]
            if originales:
                st.image(image, width="stretch", channels="BGR" if image.ndim == 3 else "RGB")
            else:
                toutes = st.checkbox("Montrer toutes les zones lues", value=False)
                st.image(commun.dessiner(image, zones(bdc, champs_vus, ech_vus, en_anomalie, no, toutes)),
                         width="stretch", caption="Bleu : sélection · orange : à vérifier · vert : lu")

    st.divider()
    nouveau = appliquer(bdc, champs, ech, client)
    changements = stockage.differences(bdc, nouveau)
    qui = commun.laborantin()
    if not qui:
        st.warning("Saisissez votre nom dans la barre latérale pour enregistrer ou valider (traçabilité).")
    if changements:
        st.caption(f"{len(changements)} modification(s) non enregistrée(s) : "
                   + ", ".join(c for c, _, _ in changements[:8]) + (" ..." if len(changements) > 8 else ""))
    b = st.columns(3)
    if b[0].button("Enregistrer les corrections", disabled=not (qui and bool(changements)), width="stretch"):
        anomalies, score = recontroler(con, d, nouveau)
        n = stockage.corriger(con, bdc_id, nouveau, qui, anomalies, score)
        if anomalies and d["statut"] == "brouillon":
            stockage.changer_statut(con, bdc_id, "a verifier", qui)
        st.toast(f"{n} correction(s) enregistrée(s)")
        st.rerun()
    anomalies_si, _ = recontroler(con, d, nouveau)
    erreurs = [a for a in anomalies_si if a["niveau"] == "erreur"]
    forcer = b[2].checkbox("Valider malgré les erreurs", value=False, disabled=not erreurs) if erreurs else False
    if b[1].button("Valider (part dans le CSV LIMS)", type="primary", disabled=bool(not qui or d["statut"] == "valide"
                                                                                 or (erreurs and not forcer)), width="stretch"):
        anomalies, score = recontroler(con, d, nouveau)
        stockage.corriger(con, bdc_id, nouveau, qui, anomalies, score)
        stockage.changer_statut(con, bdc_id, "valide", qui)
        verite.ecrire(cfg.chemin("verite"), d["fichier"], nouveau, qui)
        st.toast("BDC validé : il part dans le CSV LIMS et rejoint le jeu de référence")
        st.rerun()
    if d["statut"] == "valide" and b[2].button("Repasser « à vérifier »", disabled=not qui, width="stretch"):
        stockage.changer_statut(con, bdc_id, "a verifier", qui)
        st.rerun()
