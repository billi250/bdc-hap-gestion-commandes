"""Lecture des bons de commande par un modèle vision-langage local (Qwen3-VL via Ollama).

Le modèle lit la page entière, manuscrit compris, et renvoie directement les champs structurés au format JSON
imposé (schéma ci-dessous). Tout reste sur le poste : l'image est envoyée au serveur Ollama local
(http://localhost:11434), jamais à un service extérieur. Température 0 et graine fixe : même entrée, même sortie.
"""
import base64
import json
import urllib.error
import urllib.request

import cv2

MODELE = "qwen3-vl:8b"
URL = "http://localhost:11434"
COTE_MAX = 1600            # le modèle n'a pas besoin de plus pour lire une page A4
DELAI_MAX_S = 600

SCHEMA = {
    "type": "object",
    "properties": {
        "emetteur": {"type": "object", "properties": {k: {"type": ["string", "null"]} for k in
                                                       ("nom", "adresse", "siret", "telephone", "email")}},
        "charge_affaires": {"type": ["string", "null"]},
        "client_final": {"type": "object", "properties": {k: {"type": ["string", "null"]} for k in
                                                           ("nom", "adresse_site", "parcelles", "commune", "telephone",
                                                            "interlocuteur", "email")}},
        "laboratoire": {"type": ["string", "null"]},
        "numero_commande": {"type": ["string", "null"]},
        "date_commande": {"type": ["string", "null"]},
        "date_prelevement": {"type": ["string", "null"]},
        "delai": {"type": ["string", "null"], "enum": ["standard", "normal", "urgent", "express", None]},
        "echantillons": {"type": "array", "items": {"type": "object", "properties": {
            "reference": {"type": ["string", "null"]}, "profondeur": {"type": ["string", "null"]},
            "analyses": {"type": "array", "items": {"type": "string"}},
            "observations": {"type": ["string", "null"]}, "ligne_barree": {"type": "boolean"}},
            "required": ["reference", "analyses", "ligne_barree"]}},
    },
    "required": ["emetteur", "client_final", "numero_commande", "echantillons"],
}

CONSIGNE = """Tu lis un bon de commande d'analyses de sols envoyé à un laboratoire. Recopie exactement ce qui est
écrit (imprimé ou manuscrit), sans inventer : si une information est absente ou illisible, mets null.
Rôles :
- emetteur : l'organisme dont l'en-tête figure en haut du bon (nom, adresse, SIRET, téléphone, e-mail) ;
- charge_affaires : la personne indiquée comme « chargé d'affaires » (ou technicien) ;
- client_final : le client ou le site pour qui sont faites les analyses (« client final », « raison sociale »,
  « exploitant »), avec l'adresse du site, les parcelles, la commune, le téléphone et l'interlocuteur ;
- laboratoire : le laboratoire destinataire.
Dates au format JJ/MM/AAAA. Téléphones au format 0X XX XX XX XX. Délai : la case cochée (standard, normal,
urgent ou express), ou le délai écrit.
Tableau des échantillons : une entrée par ligne remplie, dans l'ordre ; analyses demandées sous forme de codes
(« HAP 16 », « MS », « pH », « ETM », « COT »), en lisant les cases cochées ou le texte ; ligne_barree = true si
la ligne est rayée d'un trait. Réponds uniquement par un objet JSON avec exactement ces clés :
{"emetteur": {"nom": "", "adresse": "", "siret": "", "telephone": "", "email": ""},
 "charge_affaires": "",
 "client_final": {"nom": "", "adresse_site": "", "parcelles": "", "commune": "", "telephone": "", "interlocuteur": "", "email": ""},
 "laboratoire": "", "numero_commande": "", "date_commande": "", "date_prelevement": "", "delai": "",
 "echantillons": [{"reference": "", "profondeur": "", "analyses": [], "observations": "", "ligne_barree": false}]}"""


def disponible(modele=MODELE):
    """Vrai si le serveur Ollama local répond et que le modèle est installé."""
    try:
        with urllib.request.urlopen(f"{URL}/api/tags", timeout=3) as r:
            noms = [m["name"] for m in json.load(r).get("models", [])]
        return any(n == modele or n.startswith(modele + ":") for n in noms)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _image_base64(gris):
    f = min(1.0, COTE_MAX / max(gris.shape[:2]))
    if f < 1.0:
        gris = cv2.resize(gris, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".png", gris)[1].tobytes()).decode()


ALIAS = {"raison_sociale": "nom", "raisonsociale": "nom", "ref": "reference", "réf": "reference",
         "référence": "reference", "tableau_échantillons": "echantillons", "tableau_echantillons": "echantillons",
         "échantillons": "echantillons", "délai": "delai", "chargé_affaires": "charge_affaires",
         "chargé_d_affaires": "charge_affaires", "téléphone": "telephone", "prélèvement": "date_prelevement"}


def _cles(objet):
    """Ramène les clés inventées par le modèle (accents, synonymes, majuscules) aux clés attendues ; remonte
    les dates regroupées sous une clé « dates »."""
    if isinstance(objet, list):
        return [_cles(o) for o in objet]
    if not isinstance(objet, dict):
        return objet
    sortie = {}
    for k, v in objet.items():
        cle = ALIAS.get(k.lower(), ALIAS.get(k, k.lower()))
        if cle == "dates" and isinstance(v, dict):
            sortie.update({ALIAS.get(a, a): b for a, b in v.items()})
        else:
            sortie[cle] = _cles(v)
    return sortie


def _delai(valeur):
    """Délai ramené à standard, normal, urgent ou express (« Urgent J+5 » -> urgent)."""
    t = (valeur or "").lower()
    return next((d for d in ("standard", "normal", "urgent", "express") if d in t), valeur or None)


def lire_bon(images, modele=MODELE):
    """images : pages prétraitées (niveaux de gris), dans l'ordre. Renvoie la réponse du modèle (dict).
    Qwen3-VL sous Ollama renvoie parfois le JSON dans le champ de réflexion : on lit l'un ou l'autre."""
    requete = dict(model=modele, stream=False, think=False, format="json",
                   options=dict(temperature=0, seed=0, num_ctx=8192),
                   messages=[dict(role="user", content=CONSIGNE, images=[_image_base64(i) for i in images])])
    req = urllib.request.Request(f"{URL}/api/chat", data=json.dumps(requete).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=DELAI_MAX_S) as r:
        message = json.load(r)["message"]
    texte = (message.get("content") or "").strip() or (message.get("thinking") or "").strip()
    debut, fin = texte.find("{"), texte.rfind("}")
    reponse = _cles(json.loads(texte[debut:fin + 1]))
    reponse["delai"] = _delai(reponse.get("delai"))
    return reponse


CORRESPONDANCE = {  # champ du document <- chemin dans la réponse du modèle
    "emetteur": ("emetteur", "nom"), "emetteur_adresse": ("emetteur", "adresse"),
    "emetteur_siret": ("emetteur", "siret"), "emetteur_telephone": ("emetteur", "telephone"),
    "contact_emetteur": ("charge_affaires",), "client_final": ("client_final", "nom"),
    "adresse_site": ("client_final", "adresse_site"), "parcelles": ("client_final", "parcelles"),
    "commune": ("client_final", "commune"), "client_telephone": ("client_final", "telephone"),
    "client_interlocuteur": ("client_final", "interlocuteur"), "laboratoire": ("laboratoire",),
    "numero_commande": ("numero_commande",), "date_commande": ("date_commande",),
    "date_prelevement": ("date_prelevement",), "delai": ("delai",),
}


def _chemin(j, chemin):
    for cle in chemin:
        j = (j or {}).get(cle) if isinstance(j, dict) else None
    return j.strip() if isinstance(j, str) and j.strip() else None


def champs_depuis_reponse(j):
    """Champs au format du document (valeur, confiance None, source) à partir de la réponse du modèle."""
    return {nom: (dict(valeur=v, confiance=None, ligne=f"lu par {MODELE}") if (v := _chemin(j, chemin)) else None)
            for nom, chemin in CORRESPONDANCE.items()}


def echantillons_depuis_reponse(j):
    """Lignes du tableau au format attendu par reconstruction.reconstruire (clé « barree »)."""
    return [dict(reference=e.get("reference") or "", profondeur=e.get("profondeur") or "",
                 analyses=" + ".join(e.get("analyses") or []), remarques=e.get("observations") or "",
                 barree=bool(e.get("ligne_barree"))) for e in (j or {}).get("echantillons") or []]
