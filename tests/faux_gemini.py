"""Faux client Gemini pour les tests : réponses préparées à l'avance, aucun appel réseau."""
import copy
import json
from types import SimpleNamespace

from bdc import config as cfg


class Reponse(SimpleNamespace):
    """Réponse minimale : texte, appels de fonctions, contenu, jetons."""


def texte(contenu):
    t = contenu if isinstance(contenu, str) else json.dumps(contenu, ensure_ascii=False)
    return Reponse(text=t, function_calls=None, candidates=[SimpleNamespace(content=None)],
                   usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20))


def appel(nom, **arguments):
    from google.genai import types
    contenu = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=nom, args=arguments))])
    return Reponse(text=None, function_calls=[SimpleNamespace(name=nom, args=arguments)],
                   candidates=[SimpleNamespace(content=contenu)], usage_metadata=None)


class FauxClient:
    """client.models.generate_content renvoie les réponses dans l'ordre ; une exception est levée telle quelle."""

    def __init__(self, reponses):
        self.reponses = list(reponses)
        self.requetes = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requetes.append(dict(model=model, contents=contents, config=config))
        r = self.reponses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def config_test(tmp_path):
    """Configuration réelle, chemins redirigés vers un dossier temporaire."""
    c = copy.deepcopy(cfg.charger())
    c["chemins"] = {k: str(tmp_path / v.split("/")[-1]) for k, v in c["chemins"].items()}
    return c
