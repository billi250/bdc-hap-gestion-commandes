"""Jeu de référence : BDC corrigés à la main, et score d'une extraction contre eux.

Chaque BDC validé par un laborantin est écrit dans out/verite/<fichier>.json : le jeu de test s'enrichit à
chaque validation. Un fichier de vérité peut être partiel : seuls les champs qu'il contient sont notés.
Le score sert à comparer deux modèles (ou deux versions) : on n'adopte un modèle que si son score ne
baisse pas.
"""
import json
import re
from pathlib import Path

from .champs import sans_accents
from .controles import annule
from .schema import CHAMPS

NUMERIQUES = {"emetteur_siret", "emetteur_telephone", "client_telephone", "numero_commande", "nb_echantillons_annonce"}


def normaliser(nom, valeur):
    """Forme comparable : chiffres seuls pour les numéros, sinon minuscules sans accents ni ponctuation."""
    if valeur is None:
        return ""
    if nom in NUMERIQUES:
        return re.sub(r"\D", "", str(valeur)).lstrip("0") or re.sub(r"\D", "", str(valeur))
    return re.sub(r"[^a-z0-9]+", " ", sans_accents(str(valeur))).strip()


def ecrire(dossier, fichier, bdc, qui):
    """Écrit la vérité d'un BDC validé (valeurs seules, sans confiance ni boîte). Renvoie le chemin."""
    verite = dict(fichier=fichier, valide_par=qui, partielle=False,
                  champs={n: (bdc.get(n) or {}).get("valeur") for n in CHAMPS},
                  echantillons=[dict(reference=e.get("reference"), profondeur=e.get("profondeur"),
                                     analyses=e.get("analyses") or [], annule=annule(e))
                                for e in bdc.get("echantillons") or []])
    chemin = Path(dossier) / f"{Path(fichier).name}.json"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(verite, ensure_ascii=False, indent=2), encoding="utf-8")
    return chemin


def lire_tout(dossier, filtre=None):
    """Vérités du dossier (liste de dict), filtrées par une partie du nom de fichier."""
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(Path(dossier).glob("*.json"))
            if not filtre or filtre in f.name]


def noter(bdc, verite):
    """Compare une extraction (dict BonDeCommande) à la vérité. Renvoie dict(justes, total, details)."""
    details = []
    for nom, attendu in (verite.get("champs") or {}).items():
        if attendu is None and verite.get("partielle"):
            continue
        lu = (bdc.get(nom) or {}).get("valeur")
        details.append(dict(champ=nom, attendu=attendu, lu=lu, juste=normaliser(nom, lu) == normaliser(nom, attendu)))
    lus = bdc.get("echantillons") or []
    for i, e in enumerate(verite.get("echantillons") or [], 1):
        l = lus[i - 1] if i <= len(lus) else {}
        for cle in ("reference", "profondeur"):
            if e.get(cle) is not None:
                details.append(dict(champ=f"echantillons[{i}].{cle}", attendu=e[cle], lu=l.get(cle),
                                    juste=normaliser(cle, l.get(cle)) == normaliser(cle, e[cle])))
        if "analyses" in e:
            details.append(dict(champ=f"echantillons[{i}].analyses", attendu=e["analyses"], lu=l.get("analyses"),
                                juste=sorted(map(sans_accents, e["analyses"])) == sorted(map(sans_accents, l.get("analyses") or []))))
        if "annule" in e:
            details.append(dict(champ=f"echantillons[{i}].annule", attendu=e["annule"], lu=annule(l) if l else None,
                                juste=bool(l) and annule(l) == e["annule"]))
    if verite.get("echantillons") is not None:
        details.append(dict(champ="nombre d'échantillons", attendu=len(verite["echantillons"]), lu=len(lus),
                            juste=len(verite["echantillons"]) == len(lus)))
    justes = sum(d["juste"] for d in details)
    return dict(justes=justes, total=len(details), details=details)
