"""Base SQLite : BDC, statut, historique des corrections.

Le scan original n'est jamais modifié : on stocke à côté la lecture du modèle (« extrait », figée) et la
version corrigée par le laborantin (« courant »). Chaque différence entre deux versions est écrite dans la
table des corrections : champ, valeur avant, valeur après, qui, quand, origine (agent ou laborantin).
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .controles import annule
from .schema import CHAMPS

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bdc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fichier TEXT NOT NULL, sha256 TEXT, scan TEXT, recu_le TEXT NOT NULL,
    modele TEXT, statut TEXT NOT NULL DEFAULT 'brouillon', score REAL,
    numero TEXT, emetteur TEXT, client TEXT, date_commande TEXT, delai TEXT, nb_echantillons INTEGER,
    extrait TEXT NOT NULL, courant TEXT NOT NULL, anomalies TEXT, infos TEXT,
    valide_le TEXT, valide_par TEXT
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bdc_id INTEGER NOT NULL REFERENCES bdc(id), champ TEXT NOT NULL,
    avant TEXT, apres TEXT, qui TEXT NOT NULL, quand TEXT NOT NULL, origine TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bdc_numero ON bdc(numero);
CREATE INDEX IF NOT EXISTS idx_corr_bdc ON corrections(bdc_id);
"""
COLONNES_RESUME = ("id", "fichier", "sha256", "scan", "recu_le", "modele", "statut", "score", "numero", "emetteur",
                   "client", "date_commande", "delai", "nb_echantillons", "valide_le", "valide_par")


def maintenant():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ouvrir(chemin):
    """Connexion à la base (créée si besoin)."""
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(chemin), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA_SQL)
    return con


def _resume(bdc):
    """Colonnes de recherche tirées du contenu du BDC."""
    v = lambda n: (bdc.get(n) or {}).get("valeur")
    return dict(numero=v("numero_commande"), emetteur=v("emetteur_nom"), client=v("client_nom"),
                date_commande=v("date_commande"), delai=v("delai"),
                nb_echantillons=sum(not annule(e) for e in bdc.get("echantillons") or []))


def aplatir(bdc):
    """Valeurs du BDC à plat (champ -> texte), pour comparer deux versions champ par champ."""
    plat = {n: (bdc.get(n) or {}).get("valeur") for n in CHAMPS}
    for i, e in enumerate(bdc.get("echantillons") or [], 1):
        for cle in ("reference", "profondeur", "observations"):
            plat[f"echantillons[{i}].{cle}"] = e.get(cle)
        plat[f"echantillons[{i}].analyses"] = " + ".join(e.get("analyses") or []) or None
        plat[f"echantillons[{i}].ligne_barree"] = "oui" if e.get("ligne_barree") else "non"
    return plat


def differences(avant, apres):
    """[(champ, valeur avant, valeur après)] des champs qui ont changé (ajouts et suppressions compris)."""
    a, b = aplatir(avant), aplatir(apres)
    return [(k, a.get(k), b.get(k)) for k in list(dict.fromkeys(list(a) + list(b))) if a.get(k) != b.get(k)]


def enregistrer(con, fichier, bdc, extrait, anomalies, score, statut, infos, sha256=None, scan=None,
                modele=None, relectures=()):
    """Nouveau BDC. extrait : lecture du modèle avant relecture ; bdc : après relecture par l'agent.
    Les changements de l'agent sont tracés comme des corrections d'origine « agent ». Renvoie l'id."""
    cur = con.execute(
        "INSERT INTO bdc (fichier, sha256, scan, recu_le, modele, statut, score, numero, emetteur, client, "
        "date_commande, delai, nb_echantillons, extrait, courant, anomalies, infos) VALUES "
        "(:fichier, :sha256, :scan, :recu_le, :modele, :statut, :score, :numero, :emetteur, :client, "
        ":date_commande, :delai, :nb_echantillons, :extrait, :courant, :anomalies, :infos)",
        dict(fichier=fichier, sha256=sha256, scan=scan, recu_le=maintenant(), modele=modele, statut=statut,
             score=score, extrait=json.dumps(extrait, ensure_ascii=False), courant=json.dumps(bdc, ensure_ascii=False),
             anomalies=json.dumps(anomalies, ensure_ascii=False), infos=json.dumps(infos, ensure_ascii=False, default=str),
             **_resume(bdc)))
    bdc_id = cur.lastrowid
    for r in relectures:
        if r.get("retenu"):
            con.execute("INSERT INTO corrections (bdc_id, champ, avant, apres, qui, quand, origine) VALUES "
                        "(?, ?, ?, ?, ?, ?, 'agent')", (bdc_id, r["champ"], r["avant"], r["apres"],
                                                         f"agent de relecture ({r.get('modele', '')})", maintenant()))
    con.commit()
    return bdc_id


def lire(con, bdc_id):
    """Un BDC complet (dict), contenus JSON décodés ; None s'il n'existe pas."""
    ligne = con.execute("SELECT * FROM bdc WHERE id = ?", (bdc_id,)).fetchone()
    if ligne is None:
        return None
    d = dict(ligne)
    for cle in ("extrait", "courant", "anomalies", "infos"):
        d[cle] = json.loads(d[cle]) if d[cle] else None
    return d


def lister(con, statut=None):
    """Résumé de tous les BDC (sans le contenu JSON), le plus récent d'abord."""
    sql = f"SELECT {', '.join(COLONNES_RESUME)} FROM bdc" + (" WHERE statut = ?" if statut else "") + " ORDER BY id DESC"
    return [dict(l) for l in con.execute(sql, (statut,) if statut else ())]


def corriger(con, bdc_id, nouveau, qui, anomalies=None, score=None):
    """Enregistre la version corrigée par le laborantin et trace chaque champ modifié. Renvoie le nombre de
    champs modifiés. Un BDC validé repasse « a verifier » s'il est modifié."""
    actuel = lire(con, bdc_id)
    diffs = differences(actuel["courant"], nouveau)
    if not diffs:
        return 0
    quand = maintenant()
    con.executemany("INSERT INTO corrections (bdc_id, champ, avant, apres, qui, quand, origine) VALUES "
                    "(?, ?, ?, ?, ?, ?, 'laborantin')", [(bdc_id, c, a, b, qui, quand) for c, a, b in diffs])
    maj = dict(courant=json.dumps(nouveau, ensure_ascii=False), **_resume(nouveau))
    if anomalies is not None:
        maj.update(anomalies=json.dumps(anomalies, ensure_ascii=False), score=score)
    if actuel["statut"] == "valide":
        maj.update(statut="a verifier", valide_le=None, valide_par=None)
    con.execute(f"UPDATE bdc SET {', '.join(f'{k} = :{k}' for k in maj)} WHERE id = :id", dict(maj, id=bdc_id))
    con.commit()
    return len(diffs)


def changer_statut(con, bdc_id, statut, qui):
    """Brouillon, à vérifier ou validé. Le passage de statut est tracé comme une correction."""
    avant = lire(con, bdc_id)["statut"]
    if avant == statut:
        return
    valide = statut == "valide"
    con.execute("UPDATE bdc SET statut = ?, valide_le = ?, valide_par = ? WHERE id = ?",
                (statut, maintenant() if valide else None, qui if valide else None, bdc_id))
    con.execute("INSERT INTO corrections (bdc_id, champ, avant, apres, qui, quand, origine) VALUES "
                "(?, 'statut', ?, ?, ?, ?, 'laborantin')", (bdc_id, avant, statut, qui, maintenant()))
    con.commit()


def historique(con, bdc_id):
    """Corrections d'un BDC, dans l'ordre."""
    return [dict(l) for l in con.execute("SELECT champ, avant, apres, qui, quand, origine FROM corrections "
                                         "WHERE bdc_id = ? ORDER BY id", (bdc_id,))]


def _contient(texte, motif):
    return motif.lower() in (texte or "").lower()


def chercher(con, texte=None, numero=None, client=None, date_min=None, date_max=None, statut=None,
             analyse=None, echantillon=None):
    """Recherche multicritère. texte : n'importe où (n°, émetteur, client, références, analyses). Dates au
    format JJ/MM/AAAA, comparées à la date de commande. Renvoie les résumés trouvés."""
    from .controles import lire_date
    trouves = []
    for ligne in con.execute("SELECT * FROM bdc ORDER BY id DESC"):
        d = dict(ligne)
        bdc = json.loads(d["courant"])
        ech = bdc.get("echantillons") or []
        tout = " ".join(str(v) for v in aplatir(bdc).values() if v) + " " + d["fichier"]
        dc = lire_date(d["date_commande"])
        if ((texte and not _contient(tout, texte))
                or (numero and not _contient(d["numero"], numero))
                or (client and not (_contient(d["emetteur"], client) or _contient(d["client"], client)))
                or (statut and d["statut"] != statut)
                or (date_min and (not dc or dc < lire_date(date_min)))
                or (date_max and (not dc or dc > lire_date(date_max)))
                or (analyse and not any(_contient(a, analyse) for e in ech for a in e.get("analyses") or []))
                or (echantillon and not any(_contient(e.get("reference"), echantillon) for e in ech))):
            continue
        trouves.append({k: d[k] for k in COLONNES_RESUME})
    return trouves


def statistiques(con):
    """BDC reçus, par statut, par émetteur ; délai moyen de validation (heures) par émetteur."""
    lignes = lister(con)
    par_emetteur = {}
    for l in lignes:
        e = par_emetteur.setdefault(l["emetteur"] or "(inconnu)", dict(recus=0, valides=0, en_attente=0, heures=[]))
        e["recus"] += 1
        if l["statut"] == "valide":
            e["valides"] += 1
            if l["valide_le"]:
                e["heures"].append((datetime.fromisoformat(l["valide_le"]) -
                                    datetime.fromisoformat(l["recu_le"])).total_seconds() / 3600)
        else:
            e["en_attente"] += 1
    for e in par_emetteur.values():
        h = e.pop("heures")
        e["delai_moyen_validation_h"] = round(sum(h) / len(h), 1) if h else None
    return dict(recus=len(lignes), par_statut={s: sum(l["statut"] == s for l in lignes)
                                                for s in ("brouillon", "a verifier", "valide")},
                par_emetteur=par_emetteur)
