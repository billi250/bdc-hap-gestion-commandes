"""Mesure de l'orientation détectée contre la vérité terrain établie à l'oeil (verite_terrain.json).

Usage : python evaluer.py out/preprocess_log.jsonl verite_terrain.json
"""
import argparse
import json
from pathlib import Path

from bdc.journal import lire_journal


def comparer(entrees, verite):
    """Renvoie la liste (page, détecté, attendu, confiance, juste) pour chaque entrée du journal."""
    lignes = []
    for e in entrees:
        cle = f"{Path(e['fichier_source']).stem}_p{e['page']}"
        attendu = verite.get(cle, verite["_defaut"])
        lignes.append((cle, e["angle_orientation"], attendu, e["confiance_orientation"],
                       e["angle_orientation"] == attendu))
    return lignes


def main():
    parser = argparse.ArgumentParser(description="Exactitude de l'orientation par rapport à la vérité terrain.")
    parser.add_argument("journal")
    parser.add_argument("verite")
    args = parser.parse_args()
    lignes = comparer(lire_journal(args.journal), json.loads(Path(args.verite).read_text(encoding="utf-8")))
    for cle, det, att, conf, ok in lignes:
        print(f"{cle:38s} détecté {det:3d}  attendu {att:3d}  confiance {conf:.2f}  {'OK' if ok else 'FAUX'}")
    justes = sum(l[4] for l in lignes)
    print(f"\nOrientation juste : {justes}/{len(lignes)} ; confiance minimale d'une page juste : "
          f"{min(l[3] for l in lignes if l[4]):.2f}")


if __name__ == "__main__":
    main()
