"""Journal des transformations : un enregistrement JSON par page, une ligne par enregistrement (JSONL)."""
import json
from pathlib import Path


def initialiser_journal(chemin):
    """Crée (ou vide) le fichier journal : chaque exécution repart d'un journal propre."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("", encoding="utf-8")
    return chemin


def ecrire_entree(chemin, entree):
    """Ajoute un enregistrement (dict sérialisable en JSON) à la fin du journal."""
    with open(chemin, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entree, ensure_ascii=False) + "\n")


def lire_journal(chemin):
    """Relit le journal : liste des enregistrements, dans l'ordre d'écriture."""
    with open(chemin, encoding="utf-8") as fh:
        return [json.loads(ligne) for ligne in fh if ligne.strip()]
