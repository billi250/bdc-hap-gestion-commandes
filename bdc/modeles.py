"""Passerelle unique vers les modèles (Gemini par API, ou Qwen3-VL local pour la variante RGPD).

Le reste du code ne connaît pas le fournisseur : il demande un usage (« extraction », « relecture »,
« chatbot ») et cette passerelle lit config.yaml pour savoir quel modèle appeler.

Garde-fous :
- chaque envoi à un service externe est tracé (date, usage, modèle, document, empreinte SHA-256 des images,
  justification) dans out/envois_externes.jsonl, avant l'envoi ;
- si le modèle principal est indisponible (quota, panne), on bascule sur le modèle « secours » ;
- la réponse JSON est validée par le schéma pydantic : une réponse hors schéma lève une erreur.
"""
import base64
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2

from . import config as cfg
from .journal import ecrire_entree

COTE_MAX = 2000          # une page A4 reste lisible, manuscrit compris, à 2000 pixels de haut
URL_OLLAMA = "http://localhost:11434"


class ModeleIndisponible(RuntimeError):
    """Aucun modèle n'a pu répondre (clé absente, quota, panne, serveur local arrêté)."""


def png(image, cote_max=COTE_MAX):
    """Image (niveaux de gris ou BGR) réduite si besoin et encodée en PNG."""
    f = min(1.0, cote_max / max(image.shape[:2]))
    if f < 1.0:
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", image)[1].tobytes()


def tracer(usage, reglage, document, images, justification, statut, journal=None, **autres):
    """Ajoute une ligne au journal des envois externes. Les images ne sont pas copiées : seule leur empreinte."""
    entree = dict(date=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), usage=usage,
                  fournisseur=reglage["fournisseur"], modele=reglage["modele"], document=document,
                  images_sha256=[hashlib.sha256(i).hexdigest() for i in images],
                  octets=sum(len(i) for i in images), justification=justification, statut=statut, **autres)
    chemin = Path(journal or cfg.chemin("envois_externes"))
    chemin.parent.mkdir(parents=True, exist_ok=True)
    ecrire_entree(chemin, entree)
    return entree


def lire_envois(journal=None):
    """Toutes les lignes du journal des envois externes (liste de dict), la plus récente en dernier."""
    chemin = Path(journal or cfg.chemin("envois_externes"))
    if not chemin.exists():
        return []
    return [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]


class Passerelle:
    """Appels aux modèles. client : client Gemini injecté (tests) ; sinon créé avec la clé d'API.
    locale=True : tout passe par le modèle local de config.yaml (aucun envoi externe). secours=False : pas de
    bascule (benchmark : on note le modèle demandé, pas son remplaçant)."""

    ATTENTES_S = (2, 5, 15)     # entre deux essais quand Gemini est surchargé

    def __init__(self, config=None, client=None, locale=False, journal=None, attentes=ATTENTES_S, secours=True):
        self.config = config or cfg.charger()
        self.secours = secours
        self.attentes = attentes
        self._client = client
        self.locale = locale
        self.journal = journal

    # --- Gemini -------------------------------------------------------------------------------------------
    def client(self):
        """Client Gemini, créé à la première utilisation."""
        if self._client is None:
            cle = cfg.cle_api()
            if not cle:
                raise ModeleIndisponible("Clé Gemini absente : ajouter GEMINI_API_KEY=... dans le fichier .env")
            from google import genai
            self._client = genai.Client(api_key=cle)
        return self._client

    def reglage(self, usage):
        """Réglage du modèle pour un usage (ou le modèle local en variante RGPD)."""
        return cfg.modele("locale" if self.locale else usage, self.config)

    def _config_gemini(self, reglage, schema=None, outils=None, consigne_systeme=None):
        from google.genai import types
        options = dict(seed=0, media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH)
        if reglage.get("reflexion"):
            options["thinking_config"] = types.ThinkingConfig(thinking_level=reglage["reflexion"])
        if schema is not None:
            options.update(response_mime_type="application/json", response_json_schema=schema.model_json_schema())
        if outils:
            options.update(tools=[types.Tool(function_declarations=outils)],
                           automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        if consigne_systeme:
            options["system_instruction"] = consigne_systeme
        return types.GenerateContentConfig(**options)

    def _appel_gemini(self, reglage, contenus, **options):
        from google.genai import errors
        try:
            return self.client().models.generate_content(model=reglage["modele"], contents=contenus,
                                                         config=self._config_gemini(reglage, **options))
        except errors.ClientError as e:
            if "thinking" in str(e).lower() and reglage.get("reflexion"):   # niveau non accepté par ce modèle
                return self._appel_gemini({**reglage, "reflexion": None}, contenus, **options)
            raise

    def gemini(self, usage, contenus, document="", images=(), justification="", **options):
        """Appel Gemini tracé. Surcharge passagère (503, 429) : nouvel essai après 2, 5 puis 15 s, puis bascule
        sur le modèle de secours. Renvoie (réponse brute, réglage utilisé)."""
        from google.genai import errors
        dernier = None
        for nom in (usage, "secours") if self.secours else (usage,):
            reglage = cfg.modele(nom, self.config)
            for essai, attente in enumerate((0,) + tuple(self.attentes)):
                time.sleep(attente)
                tracer(usage, reglage, document, list(images), justification, "envoi", self.journal, essai=essai + 1)
                debut = time.perf_counter()
                try:
                    reponse = self._appel_gemini(reglage, contenus, **options)
                    tracer(usage, reglage, document, [], justification, "reçu", self.journal,
                           duree_s=round(time.perf_counter() - debut, 1), jetons=_jetons(reponse))
                    return reponse, reglage
                except (errors.ServerError, errors.ClientError) as e:
                    code = getattr(e, "code", None)
                    tracer(usage, reglage, document, [], justification, f"erreur {code}", self.journal)
                    dernier = e
                    if isinstance(e, errors.ClientError) and code not in (429, 404):
                        raise ModeleIndisponible(f"Requête refusée par Gemini ({e})") from e
                    if code not in (429, 500, 503, 504):
                        break                              # modèle introuvable : passer au secours
        raise ModeleIndisponible(f"Gemini n'a pas répondu, réessayer dans quelques minutes ({dernier})")

    # --- Usage commun : image(s) + consigne -> objet validé ----------------------------------------------
    def generer_json(self, usage, consigne, images, schema, document="", justification=""):
        """Envoie les images (PNG) et la consigne, impose le schéma pydantic et renvoie (objet validé, infos)."""
        if self.locale:
            return self._ollama_json(consigne, images, schema)
        from google.genai import types
        contenus = [types.Part.from_bytes(data=i, mime_type="image/png") for i in images] + [consigne]
        reponse, reglage = self.gemini(usage, contenus, document, images, justification, schema=schema)
        objet = schema.model_validate_json(_json_seul(reponse.text))
        return objet, dict(fournisseur="gemini", modele=reglage["modele"], jetons=_jetons(reponse))

    # --- Variante locale (Ollama) ------------------------------------------------------------------------
    def _ollama_json(self, consigne, images, schema):
        reglage = cfg.modele("locale", self.config)
        requete = dict(model=reglage["modele"], stream=False, think=False, format=schema.model_json_schema(),
                       options=dict(temperature=0, seed=0, num_ctx=16384),
                       messages=[dict(role="user", content=consigne,
                                      images=[base64.b64encode(i).decode() for i in images])])
        req = urllib.request.Request(f"{URL_OLLAMA}/api/chat", data=json.dumps(requete).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                message = json.load(r)["message"]
        except OSError as e:
            raise ModeleIndisponible(f"Modèle local indisponible : lancer Ollama ({e})") from e
        texte = (message.get("content") or "").strip() or (message.get("thinking") or "").strip()
        return schema.model_validate_json(_json_seul(texte)), dict(fournisseur="ollama", modele=reglage["modele"])


def _json_seul(texte):
    """Le JSON de la réponse, sans texte ni balises autour."""
    texte = texte or ""
    debut, fin = texte.find("{"), texte.rfind("}")
    return texte[debut:fin + 1] if debut >= 0 else texte


def _jetons(reponse):
    """Nombre de jetons (entrée, sortie) d'une réponse Gemini, pour suivre le coût."""
    u = getattr(reponse, "usage_metadata", None)
    return [getattr(u, "prompt_token_count", None), getattr(u, "candidates_token_count", None)] if u else None
