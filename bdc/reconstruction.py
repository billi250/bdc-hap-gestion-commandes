"""Reconstruction du tableau des échantillons après OCR : lignes barrées (annulées), références recalées,
profondeurs et analyses normalisées. Chaque correction est tracée (valeur lue, valeur retenue, raison).

- Ligne barrée : un long trait horizontal traverse la ligne du tableau à mi-hauteur (hors traits de la
  grille). La ligne est « annulée » ; si la colonne Observations dit « annulé », c'est confirmé.
- Références : les références d'un même bon suivent un modèle (ex : ECO-4587-1, -2, -3...). Le modèle est
  reconstitué par vote, caractère par caractère, sur les références lues ; le numéro est recoupé avec le
  numéro de commande ; le dernier segment suit l'ordre des lignes.
- Profondeur : « a-b » en cm ; un premier chiffre isolé illisible (9 pour 0) est ramené à 0.
- Analyses : codes connus (HAP 16, MS, pH, ETM, COT) retrouvés malgré les erreurs d'OCR.
"""
import re
from collections import Counter
from difflib import SequenceMatcher

import cv2
import numpy as np

from .champs import sans_accents

CODES = [("HAP 16", r"h.?a.?p|ha?p ?1?6"), ("MS", r"\bm[s5$çc]\b|mat.*seche"), ("pH", r"\bp ?h\b"),
         ("ETM", r"\betm\b|metaux"), ("COT", r"\bcot\b")]
CONFUSIONS_LETTRES = {"0": "O", "1": "I", "5": "S", "€": "E", "8": "B", "6": "G", "|": "I"}
CONFUSIONS_CHIFFRES = {"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8", "G": "6", "Z": "2"}


def ligne_barree(binaire, y0, y1, x0, x1, part_min=0.3):
    """Vrai si un trait horizontal continu couvre au moins part_min de la largeur de la ligne du tableau,
    dans sa partie centrale (les traits de la grille, en haut et en bas, sont exclus)."""
    h = y1 - y0
    bande = (binaire[y0 + int(0.2 * h):y1 - int(0.2 * h), x0:x1] == 0).astype(np.uint8)
    if bande.size == 0:
        return False
    longueur = max(10, int(0.15 * (x1 - x0)))
    traits = cv2.morphologyEx(bande, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (longueur, 1)))
    return bool(traits.any(axis=0).sum() >= part_min * (x1 - x0))


def annule_par_observation(texte):
    """Vrai si les observations disent « annulé » (tolérant aux erreurs d'OCR : « anndlé », « annule »)."""
    return any(SequenceMatcher(None, mot, "annule").ratio() >= 0.7 for mot in re.findall(r"\w+", sans_accents(texte or "")))


def _vote(segments):
    """Segment majoritaire, caractère par caractère, parmi des segments de même longueur (la plus fréquente)."""
    if not segments:
        return None
    longueur = Counter(len(s) for s in segments).most_common(1)[0][0]
    pareils = [s for s in segments if len(s) == longueur]
    return "".join(Counter(s[i] for s in pareils).most_common(1)[0][0] for i in range(longueur))


def _corriger_segment(segment, chiffres):
    """Corrige les confusions d'OCR d'un segment selon qu'il doit contenir des chiffres ou des lettres."""
    table = CONFUSIONS_CHIFFRES if chiffres else CONFUSIONS_LETTRES
    return "".join(table.get(c, c) for c in segment.upper() if c.isalnum() or c in table)


def modele_references(lues, numero_commande=None, emetteur=None):
    """Modèle commun des références (liste des segments sauf le dernier) ou None s'il n'y a pas de modèle net.
    Le segment numérique proche du numéro de commande est remplacé par celui-ci (ex : 4597 -> 4587) ; un
    segment de lettres qui commence le nom de l'émetteur l'emporte sur les autres lectures (ECO / ECOTELLUS)."""
    decoupees = [re.split(r"\s*[-–_.]\s*", r.strip()) for r in lues if r and len(re.split(r"[-–_.]", r)) >= 2]
    if len(decoupees) < 2:
        return None
    nb = Counter(len(d) for d in decoupees).most_common(1)[0][0]
    decoupees = [d for d in decoupees if len(d) == nb]
    if len(decoupees) < 2:
        return None
    modele = []
    for i in range(nb - 1):
        chiffres = sum(c.isdigit() for d in decoupees for c in d[i]) > sum(c.isalpha() for d in decoupees for c in d[i])
        lectures = [_corriger_segment(d[i], chiffres) for d in decoupees]
        nom = re.sub(r"\W", "", sans_accents(emetteur or "")).upper()
        dans_nom = [s for s in lectures if not chiffres and len(s) >= 2 and nom.startswith(s)]
        seg = dans_nom[0] if dans_nom else _vote(lectures)
        numero = re.sub(r"\D", "", numero_commande or "")
        if chiffres and numero and len(seg) >= 3:
            fin = numero[-len(seg):]
            if sum(a != b for a, b in zip(seg, fin)) <= 1:
                seg = fin
        modele.append(seg)
    return modele


def reference(modele, lue, rang):
    """Référence reconstruite d'après le modèle ; le dernier segment est lu s'il est numérique, sinon déduit
    du rang de la ligne dans le tableau."""
    dernier = re.split(r"[-–_.]", lue or "")[-1]
    dernier = _corriger_segment(dernier, True) if dernier else ""
    if not dernier.isdigit():
        dernier = str(rang)
    return "-".join(modele + [dernier])


def profondeur(lue):
    """Profondeur normalisée « a-b » (cm), ou la valeur lue si elle n'a pas cette forme."""
    texte = "".join(CONFUSIONS_CHIFFRES.get(c, c) for c in (lue or ""))
    m = re.search(r"(\d{1,3})\s*[-–]\s*(\d{1,3})", texte)
    if not m:
        return lue
    debut, fin = int(m.group(1)), int(m.group(2))
    if debut != 0 and debut < 10 and fin <= 30:        # « 9-30 » : le 0 initial mal lu
        debut = 0
    return f"{debut}-{fin}"


def analyses(lue):
    """Codes d'analyses reconnus dans le texte lu (liste ordonnée, sans doublon)."""
    t = sans_accents(lue or "")
    return [code for code, motif in CODES if re.search(motif, t)]


def reconstruire(lignes, numero_commande=None, emetteur=None):
    """Lignes du tableau reconstruites : références, profondeur, analyses, statut (« a analyser » ou « annule »)
    et liste des corrections. Les lignes doivent porter la clé « barree » (bool)."""
    modele = modele_references([l.get("reference", "") for l in lignes if not l.get("barree")], numero_commande,
                               emetteur)
    sortie = []
    for rang, l in enumerate(lignes, start=1):
        nouvelle, corrections = dict(l), []
        if modele and "reference" in l:
            ref = reference(modele, l["reference"], rang)
            if ref != l["reference"]:
                corrections.append(f"référence {l['reference']!r} -> {ref!r} (modèle du bon)")
            nouvelle["reference"] = ref
        if l.get("profondeur"):
            p = profondeur(l["profondeur"])
            if p != l["profondeur"]:
                corrections.append(f"profondeur {l['profondeur']!r} -> {p!r}")
            nouvelle["profondeur"] = p
        if "analyses" in l:
            codes = analyses(l["analyses"])
            if codes:
                texte = " + ".join(codes)
                if texte != l["analyses"]:
                    corrections.append(f"analyses {l['analyses']!r} -> {texte!r}")
                nouvelle["analyses"] = texte
                for code in codes:
                    nouvelle[code.lower().replace(" ", "")] = True
        cle_obs = next((k for k in ("remarques", "observations") if k in l), None)
        obs = l.get(cle_obs, "") if cle_obs else ""
        if modele and obs:
            recalee = re.sub(r"[A-Z€0-9]{2,5}\s?[-–]\s?\d{2,6}\s?[-–]\s?\w{1,3}",
                             lambda m: reference(modele, m.group(0), rang), obs)
            if recalee != obs:
                corrections.append(f"observations {obs!r} -> {recalee!r} (référence citée)")
                nouvelle[cle_obs] = recalee
        barree, annule_obs = bool(l.get("barree")), annule_par_observation(obs)
        if barree or annule_obs:
            nouvelle["statut"] = "annule"
            nouvelle["motif_statut"] = ("ligne barrée, confirmé par les observations" if barree and annule_obs else
                                        "ligne barrée" if barree else "observations : annulé (ligne non barrée)")
        else:
            nouvelle["statut"] = "a analyser"
            nouvelle["motif_statut"] = ""
        nouvelle["corrections"] = corrections
        nouvelle.pop("barree", None)
        sortie.append(nouvelle)
    return sortie
