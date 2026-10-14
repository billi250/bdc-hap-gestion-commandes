"""Export JSON et CSV LIMS des seuls BDC validés.

CSV LIMS : une ligne par analyse à réaliser (BDC x échantillon non annulé x analyse), avec les codes LIMS
du référentiel. Séparateur « ; » et encodage UTF-8 avec BOM : le fichier s'ouvre tel quel dans Excel.
"""
import csv
import io
import json

from .controles import annule
from .extraction_llm import code_analyse

COLONNES_LIMS = ["n_bdc_interne", "numero_commande", "code_client_lims", "client", "client_final",
                 "date_commande", "date_prelevement", "delai", "reference_echantillon", "profondeur_cm",
                 "code_analyse_lims", "analyse", "observations", "valide_par", "valide_le"]


def code_client(nom, referentiel):
    """Code LIMS d'un client du référentiel ; vide si inconnu."""
    return next((c["code_lims"] for c in referentiel.get("clients", []) if c["nom"] == nom), "")


def code_lims(code, referentiel):
    """Code LIMS d'une analyse ; le code lu si l'analyse est inconnue du référentiel."""
    connu = code_analyse(code, referentiel)
    return next((a["code_lims"] for a in referentiel["analyses"] if a["code"] == connu), code)


def lignes_lims(bdc_valides, referentiel):
    """bdc_valides : BDC complets (stockage.lire) au statut « valide ». Renvoie les lignes du CSV (dict)."""
    lignes = []
    for d in bdc_valides:
        if d["statut"] != "valide":
            continue
        b = d["courant"]
        v = lambda n: (b.get(n) or {}).get("valeur") or ""
        for e in b.get("echantillons") or []:
            if annule(e):
                continue
            for analyse in e.get("analyses") or []:
                lignes.append(dict(n_bdc_interne=d["id"], numero_commande=v("numero_commande"),
                                   code_client_lims=code_client(v("emetteur_nom"), referentiel),
                                   client=v("emetteur_nom"), client_final=v("client_nom"),
                                   date_commande=v("date_commande"), date_prelevement=v("date_prelevement"),
                                   delai=v("delai"), reference_echantillon=e.get("reference") or "",
                                   profondeur_cm=e.get("profondeur") or "", code_analyse_lims=code_lims(analyse, referentiel),
                                   analyse=analyse, observations=e.get("observations") or "",
                                   valide_par=d["valide_par"] or "", valide_le=d["valide_le"] or ""))
    return lignes


def csv_lims(bdc_valides, referentiel):
    """Contenu du CSV LIMS (octets UTF-8 avec BOM)."""
    tampon = io.StringIO()
    w = csv.DictWriter(tampon, fieldnames=COLONNES_LIMS, delimiter=";")
    w.writeheader()
    w.writerows(lignes_lims(bdc_valides, referentiel))
    return ("﻿" + tampon.getvalue()).encode("utf-8")


def json_bdc(bdc_valides):
    """Contenu JSON des BDC validés (version corrigée, statut, validation)."""
    return json.dumps([dict(id=d["id"], fichier=d["fichier"], statut=d["statut"], valide_par=d["valide_par"],
                            valide_le=d["valide_le"], bdc=d["courant"]) for d in bdc_valides if d["statut"] == "valide"],
                      ensure_ascii=False, indent=2).encode("utf-8")
