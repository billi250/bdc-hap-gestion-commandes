"""Test de non-régression des modèles sur les BDC corrigés à la main (out/verite/).

Usage :
    python benchmark.py --document BDC_06_ecotellus_liasse_n1      (un seul document)
    python benchmark.py                                            (tous les BDC de référence)
    python benchmark.py --modele gemini-3.6-flash                  (essai d'une autre version)
    python benchmark.py --locale                                   (modèle local, sans envoi externe)

Pour chaque BDC de référence : prétraitement, lecture par le modèle, agent de relecture, puis comptage des
champs justes. Le résultat est écrit dans out/benchmark/ et comparé au meilleur essai précédent sur les
mêmes documents : une nouvelle version n'est adoptée que si son score ne baisse pas.
"""
import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from bdc import agent_relecture, controles, extraction_llm, verite
from bdc import config as cfg
from bdc.modeles import ModeleIndisponible, Passerelle, lire_envois
from bdc.traitement import pretraiter

DOSSIERS_SCANS = [Path("data/BDC")]


def trouver_scan(fichier, dossiers):
    """Chemin du scan d'origine d'après son nom de fichier."""
    for d in dossiers:
        if (Path(d) / fichier).exists():
            return Path(d) / fichier
    return None


def evaluer_document(v, chemin, passerelle, referentiel, seuils):
    """Extraction complète d'un document et note contre sa vérité."""
    pages = [p["gris"] for p in pretraiter(chemin.read_bytes(), chemin.name)]
    bdc_obj, infos = extraction_llm.extraire_champs(pages, passerelle, referentiel, document=chemin.name)
    bdc = bdc_obj.model_dump()
    note_brute = verite.noter(copy.deepcopy(bdc), v)
    _, journal = agent_relecture.relire(bdc, pages, passerelle, referentiel, seuils, document=chemin.name,
                                        controle=lambda b: controles.controler(b, referentiel, seuils))
    note = verite.noter(bdc, v)
    return dict(document=chemin.name, justes=note["justes"], total=note["total"],
                justes_sans_agent=note_brute["justes"], erreurs=[d for d in note["details"] if not d["juste"]],
                relectures=[e["message"] for e in journal if e["type"] == "action"])


def meilleur_precedent(dossier, documents):
    """Meilleur essai précédent portant exactement sur les mêmes documents, ou None."""
    essais = [json.loads(f.read_text(encoding="utf-8")) for f in Path(dossier).glob("*.json")]
    pareils = [e for e in essais if sorted(r["document"] for r in e["resultats"]) == sorted(documents)]
    return max(pareils, key=lambda e: e["score"], default=None)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--document", help="ne traiter que les vérités dont le nom contient ce texte")
    p.add_argument("--modele", help="identifiant de modèle à essayer à la place de celui de config.yaml")
    p.add_argument("--locale", action="store_true", help="modèle local (aucun envoi externe)")
    p.add_argument("--verite", default=None, help="dossier des vérités (défaut : config.yaml)")
    a = p.parse_args()
    config = copy.deepcopy(cfg.charger())
    if a.modele:
        config["extraction"]["modele"] = config["relecture"]["modele"] = a.modele
    referentiel, seuils = cfg.referentiel(), config["seuils"]
    passerelle = Passerelle(config, locale=a.locale, secours=False)
    verites = verite.lire_tout(a.verite or cfg.chemin("verite", config), a.document)
    if not verites:
        sys.exit("Aucune vérité trouvée : validez un BDC dans l'interface, ou vérifiez --document.")
    resultats = []
    for v in verites:
        chemin = trouver_scan(v["fichier"], DOSSIERS_SCANS + [cfg.chemin("scans", config)])
        if chemin is None:
            print(f"  scan introuvable : {v['fichier']}")
            continue
        avant, debut = len(lire_envois()), time.perf_counter()
        try:
            r = evaluer_document(v, chemin, passerelle, referentiel, seuils)
        except ModeleIndisponible as e:
            r = dict(document=chemin.name, justes=0, total=None, justes_sans_agent=0, erreurs=[], relectures=[],
                     echec=str(e)[:200])
        r["duree_s"] = round(time.perf_counter() - debut, 1)
        r["erreurs_api"] = sum(e["statut"].startswith("erreur") for e in lire_envois()[avant:])
        resultats.append(r)
        if r.get("echec"):
            print(f"  {r['document']} : ÉCHEC ({r['erreurs_api']} erreurs API, {r['duree_s']} s) {r['echec']}")
            continue
        print(f"  {r['document']} : {r['justes']}/{r['total']} justes (sans agent : {r['justes_sans_agent']}), "
              f"{r['duree_s']} s, {r['erreurs_api']} erreur(s) API")
        for e in r["erreurs"]:
            print(f"      {e['champ']} : attendu « {e['attendu']} », lu « {e['lu']} »")
    if not resultats:
        sys.exit("Aucun document évalué.")
    total = sum(r["total"] or 0 for r in resultats) + sum(1 for r in resultats if r.get("echec"))
    score = round(100 * sum(r["justes"] for r in resultats) / max(total, 1), 1)
    modele = passerelle.reglage("extraction")["modele"]
    sortie = cfg.chemin("verite", config).parent / "benchmark"
    precedent = meilleur_precedent(sortie, [r["document"] for r in resultats]) if sortie.exists() else None
    sortie.mkdir(parents=True, exist_ok=True)
    essai = dict(date=datetime.now().astimezone().isoformat(timespec="seconds"), modele=modele, score=score,
                 resultats=resultats)
    (sortie / f"{datetime.now():%Y%m%d_%H%M%S}_{modele.replace(':', '_')}.json").write_text(
        json.dumps(essai, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nScore {modele} : {score} % de champs justes sur {len(resultats)} document(s)")
    if precedent:
        verdict = "à adopter" if score >= precedent["score"] else "à NE PAS adopter (le score baisse)"
        print(f"Meilleur essai précédent : {precedent['modele']} {precedent['score']} % -> {verdict}")


if __name__ == "__main__":
    main()
