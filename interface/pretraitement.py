"""Page « Prétraitement (local) » : prétraitement (brique 1) et extraction par OCR local (Tesseract, Qwen3-VL)
de n'importe quel bon de commande, PDF ou image. Le document est traité en mémoire sur ce poste : rien
n'est envoyé à un service externe ni écrit sur le disque. Sert à diagnostiquer un scan difficile.
"""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from bdc import doublons, vlm
from bdc.extraction import extraire_document
from bdc.normalisation import PARAMS_DEFAUT
from bdc.ocr import disponible as ocr_disponible
from bdc.pdf_io import EXTENSIONS_IMAGE, lister_pdf, pages_depuis_octets
from preprocess import pretraiter_page

FORMATS = ["pdf"] + sorted(e.lstrip(".") for e in EXTENSIONS_IMAGE)
DOSSIER_EXEMPLES = Path("data/BDC")
DOSSIER_DEJA_TRAITES = Path("out/preprocessed")
VERDICTS = {"pret pour OCR": "Prêt pour l'OCR", "a verifier": "À vérifier", "a rescanner": "À rescanner"}

@st.cache_data(show_spinner=False, max_entries=20)
def traiter(octets, nom, params_json):
    """Prétraite toutes les pages d'un document. Renvoie une liste de dict (avant, gris, binaire, infos).
    params_json vide : réglages adaptés automatiquement au défaut diagnostiqué ; sinon réglages manuels."""
    params = json.loads(params_json) if params_json else None
    resultats = []
    for page in pages_depuis_octets(octets, nom):
        debut = time.perf_counter()
        propre, binaire, infos = pretraiter_page(page["image"], params, dpi=page["dpi_x"], adapter=params is None)
        infos = dict(fichier_source=nom, sha256_source=hashlib.sha256(octets).hexdigest(), page=page["page"],
                     mode_lecture=page["mode"], dpi=round(page["dpi_x"], 1) if page["dpi_x"] else None,
                     **infos, duree_s=round(time.perf_counter() - debut, 3))
        resultats.append(dict(avant=page["image"], gris=propre, binaire=binaire, infos=infos))
    return resultats


@st.cache_data(show_spinner=False, max_entries=20)
def extraire(cle, moteur, _pages):
    """Extraction (lecture locale) d'un document déjà prétraité. cle identifie le document et ses réglages."""
    return extraire_document(_pages, moteur)


CHAMPS_AFFICHES = [  # (rôle, champ)
    ("Émetteur", "emetteur"), ("Émetteur", "emetteur_adresse"), ("Émetteur", "emetteur_siret"),
    ("Émetteur", "emetteur_telephone"), ("Émetteur", "contact_emetteur"),
    ("Client final", "client_final"), ("Client final", "adresse_site"), ("Client final", "parcelles"),
    ("Client final", "commune"), ("Client final", "client_telephone"), ("Client final", "client_interlocuteur"),
    ("Laboratoire", "laboratoire"), ("Commande", "numero_commande"), ("Commande", "date_commande"),
    ("Commande", "date_prelevement"), ("Commande", "delai")]


def afficher_extraction(nom, resultats, cle, moteur):
    """Champs et échantillons extraits, modifiables avant export (la personne corrige ce que l'OCR a mal lu)."""
    doc = extraire(cle, moteur, [(r["gris"], r["binaire"]) for r in resultats])
    st.markdown(f"#### Extraction : {nom}")
    for a in doc["alertes"]:
        st.warning(a)
    for r in doc["remarques"]:
        st.caption(r)
    champs = [dict(role=r, champ=c, valeur=(doc["champs"].get(c) or {}).get("valeur", ""),
                   confiance=(doc["champs"].get(c) or {}).get("confiance"),
                   autre_lecture=(doc["champs"].get(c) or {}).get("autre_lecture") or (doc["champs"].get(c) or {}).get("lue", ""),
                   source=(doc["champs"].get(c) or {}).get("source", "")) for r, c in CHAMPS_AFFICHES]
    for v in doc["champs"].get("emails") or []:
        champs.append(dict(role="Contact", champ="email", valeur=v["valeur"], confiance=v["confiance"],
                           autre_lecture="", source=""))
    st.caption("Champs par rôle (modifiables). « autre_lecture » : la lecture de l'autre moteur (ou la valeur lue "
               "avant recalage sur le référentiel). Confiance 95 : lectures identiques ; 55 : lectures différentes.")
    corriges = st.data_editor(pd.DataFrame(champs), hide_index=True, width="stretch", key=f"champs_{cle}",
                              disabled=["role", "champ", "autre_lecture", "confiance", "source"], num_rows="dynamic")
    st.caption(f"Tableau des échantillons reconstruit ({len(doc['echantillons'])} lignes, confiance OCR "
               f"{doc['confiance_tableau'] if doc['confiance_tableau'] is not None else '-'}). Les lignes barrées "
               "sont « annule » ; la colonne « corrections » trace chaque valeur recalée.")
    tableau = pd.DataFrame([{**e, "corrections": " ; ".join(e.get("corrections", []))} for e in doc["echantillons"]])
    if "statut" in tableau:
        tableau = tableau[["statut"] + [c for c in tableau.columns if c != "statut"]]
    echantillons = st.data_editor(tableau, hide_index=True, width="stretch", key=f"ech_{cle}", num_rows="dynamic",
                                  column_config={"statut": st.column_config.SelectboxColumn(
                                      "statut", options=["a analyser", "annule"], required=True)})
    export = dict(document=nom, champs=corriges.to_dict("records"), echantillons=echantillons.to_dict("records"),
                  alertes=doc["alertes"], remarques=doc["remarques"])
    st.download_button("Télécharger l'extraction corrigée (JSON)", json.dumps(export, ensure_ascii=False, indent=2,
                       default=str), f"{Path(nom).stem}_extraction.json", mime="application/json", key=f"dl_{cle}")
    with st.expander("Texte OCR brut"):
        st.text("\n\n".join(doc["texte_ocr"]))


def png(image):
    """Encode une image en PNG (octets)."""
    return cv2.imencode(".png", image)[1].tobytes()


def archive(documents):
    """Zip des pages prétraitées (binaire et gris) et du journal JSONL de tous les documents."""
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w", zipfile.ZIP_DEFLATED) as z:
        journal = []
        for nom, resultats in documents.items():
            for r in resultats:
                base = f"{Path(nom).stem}_p{r['infos']['page']}"
                z.writestr(f"preprocessed/{base}.png", png(r["binaire"]))
                z.writestr(f"preprocessed/{base}_gris.png", png(r["gris"]))
                journal.append(json.dumps(r["infos"], ensure_ascii=False))
        z.writestr("preprocess_log.jsonl", "\n".join(journal) + "\n")
    return tampon.getvalue()


@st.cache_data(show_spinner="Chargement des documents déjà traités...")
def deja_traites(dossier):
    """Pages binaires déjà prétraitées (out/preprocessed) et leurs empreintes, pour repérer les doublons."""
    pages = {f.stem: cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in sorted(Path(dossier).glob("*_p[0-9]*.png"))
             if not f.stem.endswith("_gris")}
    return pages, {k: doublons.empreinte(v) for k, v in pages.items()}


def chercher_doublons(traites, comparer_au_corpus):
    """Paires ressemblantes entre les pages déposées, et avec les pages déjà traitées si demandé."""
    nouvelles = {f"{Path(nom).stem}_p{r['infos']['page']} (déposé)": r["binaire"]
                 for nom, res in traites.items() for r in res}
    empreintes = {k: doublons.empreinte(v) for k, v in nouvelles.items()}
    reference, emp_ref = deja_traites(DOSSIER_DEJA_TRAITES) if comparer_au_corpus else ({}, {})
    paires = []
    ids = sorted(nouvelles)
    for i, a in enumerate(ids):
        cibles = [(b, nouvelles[b], empreintes[b]) for b in ids[i + 1:]]
        cibles += [(b, img, emp_ref[b]) for b, img in reference.items() if b != a.split(" (")[0]]
        for b, img, e in cibles:
            niveau, d, ecart = doublons.comparer(nouvelles[a], img, empreintes[a], e)
            if niveau:
                paires.append(dict(page=a, ressemble_a=b, niveau=niveau, distance_empreinte=d,
                                   ecart_contenu=None if ecart is None else round(ecart, 4)))
    ordre = {"quasi identique": 0, "a verifier": 1, "meme modele": 2}
    return sorted(paires, key=lambda p: (ordre[p["niveau"]], p["ecart_contenu"] or 1))


def barre_laterale():
    """Réglages : adaptés automatiquement au défaut diagnostiqué, ou manuels. Renvoie les paramètres
    (None = automatiques) et l'option de comparaison aux documents déjà traités."""
    st.sidebar.header("Réglages du prétraitement")
    comparer = st.sidebar.checkbox("Chercher les doublons parmi les documents déjà traités",
                                   value=DOSSIER_DEJA_TRAITES.is_dir(), disabled=not DOSSIER_DEJA_TRAITES.is_dir(),
                                   help="Compare aux pages de out/preprocessed (produites par run_all.py).")
    manuel = st.sidebar.toggle("Réglages manuels", value=False,
                               help="Par défaut, les réglages sont choisis selon le défaut diagnostiqué "
                                    "(fax, photocopie, encre pâle...).")
    if not manuel:
        st.sidebar.caption("Réglages adaptés automatiquement au défaut de chaque page.")
        st.sidebar.divider()
        st.sidebar.caption("Traitement local : le document reste sur ce poste, en mémoire.")
        return None, comparer
    st.sidebar.caption("Valeurs de départ validées sur les 28 bons de commande du corpus. "
                       "Tous les réglages utilisés sont écrits dans le journal.")
    p = dict(PARAMS_DEFAUT)
    p["debruitage_h"] = st.sidebar.slider("Débruitage (force)", 0, 30, p["debruitage_h"])
    p["bloc_binarisation"] = st.sidebar.slider("Taille du seuil local (pixels, impair)", 11, 101,
                                               p["bloc_binarisation"], step=2)
    p["constante_binarisation"] = st.sidebar.slider("Constante du seuil local", 0, 40, p["constante_binarisation"])
    p["seuil_aplat"] = st.sidebar.slider("Seuil des aplats noirs", 0, 200, p["seuil_aplat"],
                                         help="Pixel plus sombre que ce seuil = encre (bandeaux de titre).")
    p["aire_point_max"] = st.sidebar.slider("Bruit : aire max d'un point isolé (pixels)", 0, 30, p["aire_point_max"],
                                            help="0 = aucun nettoyage. Au-delà de 6, la ponctuation risque d'être effacée.")
    st.sidebar.divider()
    st.sidebar.caption("Traitement local : le document reste sur ce poste, en mémoire.")
    return p, comparer


def documents_a_traiter():
    """Documents téléversés, ou un exemple du corpus local. Renvoie [(nom, octets)]."""
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


def afficher_page(r):
    """Une page : indicateurs, alertes, avant / après, journal."""
    i = r["infos"]
    q = i["qualite"]
    c = st.columns(6)
    c[0].metric("Défaut diagnostiqué", i["diagnostic"])
    c[1].metric("Qualité", f"{q['score']:.0f} / 100", help=VERDICTS[q["verdict"]])
    c[2].metric("Orientation corrigée", f"{i['angle_orientation']}°", help=f"confiance {i['confiance_orientation']:.2f}")
    c[3].metric("Redressement", f"{i['angle_redressement']:+.2f}°")
    c[4].metric("Perspective", i["perspective"].replace("non necessaire", "inutile"))
    c[5].metric("Durée", f"{i['duree_s']:.2f} s")
    message = f"{VERDICTS[q['verdict']]}" + (f" : {', '.join(q['raisons'])}" if q["raisons"] else "")
    {"pret pour OCR": st.success, "a verifier": st.warning, "a rescanner": st.error}[q["verdict"]](message)
    if i["reglages_adaptes"]:
        st.caption(f"Réglages adaptés au défaut « {i['diagnostic']} » : {i['reglages_adaptes']}")
    for alerte in i["alertes"]:
        if not alerte.startswith("qualite"):
            st.warning(f"À vérifier : {alerte}")
    gauche, droite = st.columns(2)
    gauche.image(r["avant"], caption="Avant (page d'origine)", width="stretch",
                 channels="BGR" if r["avant"].ndim == 3 else "RGB")
    vue = droite.radio("Après", ["binaire", "gris normalisé"], horizontal=True, key=f"vue_{i['fichier_source']}_{i['page']}")
    droite.image(r["binaire"] if vue == "binaire" else r["gris"], caption=f"Après ({vue})", width="stretch")
    with st.expander("Journal de la page (JSON)"):
        st.json(i)


def page():
    st.title("Prétraitement et lecture locale")
    st.caption("Perspective (photos), orientation 0 / 90 / 180 / 270, redressement fin, puis normalisation : "
               "débruitage, retrait des ombres et taches, binarisation locale.")
    params, comparer_au_corpus = barre_laterale()
    docs = documents_a_traiter()
    if not docs:
        st.info("Déposez un document pour lancer le prétraitement.")
        return
    traites = {}
    for nom, octets in docs:
        with st.spinner(f"Prétraitement de {nom}..."):
            try:
                traites[nom] = traiter(octets, nom, json.dumps(params, sort_keys=True) if params else "")
            except ValueError as e:
                st.error(f"{nom} : {e}")
    if not traites:
        return
    synthese = pd.DataFrame([{k: r["infos"][k] for k in ("fichier_source", "page", "diagnostic", "angle_orientation",
                             "angle_redressement", "perspective")}
                             | {"qualite": r["infos"]["qualite"]["score"],
                                "verdict": VERDICTS[r["infos"]["qualite"]["verdict"]],
                                "alertes": ", ".join(r["infos"]["alertes"]) or "-"}
                             for res in traites.values() for r in res])
    st.subheader(f"Synthèse : {len(synthese)} page(s), {int((synthese.alertes != '-').sum())} avec alerte")
    st.dataframe(synthese, hide_index=True, width="stretch")
    st.download_button("Télécharger les pages prétraitées et le journal (zip)", archive(traites),
                       "pretraitement_bdc.zip", mime="application/zip")
    with st.spinner("Recherche des doublons..."):
        paires = chercher_doublons(traites, comparer_au_corpus)
    identiques = [p for p in paires if p["niveau"] == "quasi identique"]
    a_verifier = [p for p in paires if p["niveau"] == "a verifier"]
    if identiques:
        st.error(f"Doublon : {len(identiques)} page(s) quasi identique(s) à un document déjà présent "
                 "(même bon reçu deux fois, ou « annule et remplace »).")
    elif a_verifier:
        st.warning(f"Ressemblance forte avec {len(a_verifier)} page(s) : même document rescanné, ou même formulaire "
                   "peu rempli. À comparer à la main (numéro de commande, date).")
    else:
        st.success("Aucun doublon : aucune page quasi identique à un autre document"
                   + (" déjà traité." if comparer_au_corpus else " déposé."))
    if paires:
        with st.expander(f"Ressemblances ({len(paires)}) : quasi identique, à vérifier ou même modèle de formulaire"):
            st.dataframe(pd.DataFrame(paires), hide_index=True, width="stretch")
    st.divider()
    if ocr_disponible():
        if st.toggle("Extraire les champs et le tableau des échantillons (lecture locale)", value=False):
            moteurs = {"Tesseract (rapide, environ 4 s)": "tesseract"}
            if vlm.disponible():
                moteurs = {f"Les deux, comparés (Qwen3-VL + Tesseract, environ 30 s par page)": "les deux",
                           f"Qwen3-VL seul ({vlm.MODELE}, le plus fort sur le manuscrit)": "vlm", **moteurs}
            moteur = moteurs[st.radio("Moteur de lecture", list(moteurs), horizontal=True)]
            for nom, resultats in traites.items():
                with st.spinner(f"Extraction de {nom} ({moteur})..."):
                    cle = hashlib.sha256((nom + json.dumps(params, sort_keys=True, default=str)).encode()).hexdigest()[:16]
                    afficher_extraction(nom, resultats, cle, moteur)
    else:
        st.info("Extraction indisponible : installer Tesseract (brew install tesseract tesseract-lang).")
    for nom, resultats in traites.items():
        for r in resultats:
            st.divider()
            st.markdown(f"#### {nom} : page {r['infos']['page']} / {len(resultats)}")
            afficher_page(r)
