"""Journal : écriture JSONL, relecture identique, remise à zéro à chaque exécution."""
import json

from bdc.journal import ecrire_entree, initialiser_journal, lire_journal


def test_journal_bien_ecrit(tmp_path):
    chemin = initialiser_journal(tmp_path / "sous" / "log.jsonl")
    entrees = [dict(fichier_source="a.pdf", page=1, angle_orientation=90, perspective="non necessaire",
                    alertes=[]),
               dict(fichier_source="b.pdf", page=2, angle_orientation=0, perspective="appliquee",
                    alertes=["orientation incertaine"])]
    for e in entrees:
        ecrire_entree(chemin, e)
    lignes = chemin.read_text(encoding="utf-8").splitlines()
    assert len(lignes) == 2 and all(json.loads(l) for l in lignes)
    assert lire_journal(chemin) == entrees


def test_initialiser_vide_le_journal(tmp_path):
    chemin = initialiser_journal(tmp_path / "log.jsonl")
    ecrire_entree(chemin, {"page": 1})
    initialiser_journal(chemin)
    assert lire_journal(chemin) == []
