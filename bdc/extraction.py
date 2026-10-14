"""Brique 3 (première version) : extraction des champs d'en-tête, des cases cochées et du tableau des
échantillons d'un bon de commande prétraité.

Entrées : l'image grise normalisée (pour l'OCR) et l'image binaire (pour la grille et les marques) de chaque
page. Sortie : un dict par document (champs avec confiance, échantillons, alertes), sérialisable en JSON.
"""
import re

import cv2
import numpy as np

from . import vlm
from .champs import DELAIS, LIBELLE_DELAI, REFERENTIEL, extraire_champs, luhn, recaler, sans_accents
from .fusion import fusionner_champs
from .ocr import lignes_eparses, lire_cellule
from .reconstruction import ligne_barree, reconstruire
from .tableau import cellules, detecter_tableau

COLONNES = [  # nom normalisé, motif sur l'en-tête lu (sans accents, minuscules)
    ("reference", r"ref|n.? ?ech|point|echantillon"), ("profondeur", r"prof"), ("hap16", r"hap"),
    ("ms", r"^ms\b|mati"), ("ph", r"^ph\b"), ("etm", r"etm|metaux"), ("cot", r"^cot\b"),
    ("analyses", r"analyse"), ("remarques", r"remarq|observ"), ("parcelle", r"parcel|ilot"),
    ("commune", r"commune"), ("adherent", r"adherent|exploitant"), ("culture", r"culture"),
    ("type_sol", r"type|matrice"), ("position", r"position"), ("date_prelevement", r"date"),
    ("latitude", r"latitude"), ("longitude", r"longitude"), ("numero", r"^n.?$|^no$"),
]
COLONNES_COCHE = {"hap16", "ms", "ph", "etm", "cot"}
MARQUES = {"x", "✓", "v", "/", "+", "xx", "y"}
CONFIANCE_TABLEAU_MIN = 40   # en dessous, le tableau est signalé comme peu lisible
MARQUE_MIN = 25          # pixels d'encre (hors bords) au-delà desquels une case de tableau est marquée (X, coche)


def nom_colonne(entete, deja):
    """Nom normalisé d'une colonne d'après son en-tête lu par l'OCR (unique dans le tableau)."""
    t = sans_accents(entete).strip()
    for nom, motif in COLONNES:
        if re.search(motif, t) and nom not in deja:
            return nom
    base = re.sub(r"\W+", "_", t).strip("_") or "colonne"
    nom, k = base, 2
    while nom in deja:
        nom, k = f"{base}_{k}", k + 1
    return nom


def part_encre(binaire, rect, retrait=4):
    """Part de pixels d'encre dans une cellule, bords exclus (traits de la grille)."""
    x0, y0, x1, y1 = rect
    zone = binaire[y0 + retrait:y1 - retrait, x0 + retrait:x1 - retrait]
    return float((zone == 0).mean()) if zone.size else 0.0


def pixels_encre(binaire, rect, retrait=4):
    """Nombre de pixels d'encre dans une cellule, bords exclus : un X ou une coche en compte plusieurs dizaines."""
    x0, y0, x1, y1 = rect
    return int((binaire[y0 + retrait:y1 - retrait, x0 + retrait:x1 - retrait] == 0).sum())


def lire_tableau(gris, binaire):
    """Tableau des échantillons : liste de lignes (dict colonne -> valeur) et confiance moyenne des cellules.
    Les colonnes d'analyses (HAP 16, MS, pH, ETM, COT) sont lues comme des cases : marquée ou non."""
    trouve = detecter_tableau(binaire)
    if trouve is None:
        return [], None, [], None
    _, ys, xs = trouve
    grille = cellules(ys, xs)
    noms = []
    for rect in grille[0]:
        noms.append(nom_colonne(lire_cellule(gris, rect, plusieurs_lignes=True)[0], noms))
    lignes, confiances = [], []
    for i, rangee in enumerate(grille[1:], start=1):
        ligne = {"barree": ligne_barree(binaire, ys[i], ys[i + 1], xs[0], xs[-1])}
        for nom, rect in zip(noms, rangee):
            if nom in COLONNES_COCHE:
                ligne[nom] = pixels_encre(binaire, rect) >= MARQUE_MIN
            elif part_encre(binaire, rect) > 0.005:
                texte, conf = lire_cellule(gris, rect)
                ligne[nom] = texte
                confiances.append(conf)
            else:
                ligne[nom] = ""
        if any(v for k, v in ligne.items() if k != "barree"):
            lignes.append(ligne)
    return (colonnes_de_marques(lignes, noms), (round(float(np.mean(confiances)), 1) if confiances else None), noms,
            ys[0])


def colonnes_de_marques(lignes, noms):
    """Une colonne dont toutes les valeurs non vides sont des marques (x, coche) est une colonne de cases,
    même si son en-tête n'a pas été lu : ses valeurs deviennent vrai / faux."""
    for nom in noms:
        valeurs = [sans_accents(str(l.get(nom, ""))).strip(" .,|*'\"“”-") for l in lignes]
        remplies = [v for v in valeurs if v]
        if remplies and nom not in COLONNES_COCHE and all(v in MARQUES for v in remplies):
            for l, v in zip(lignes, valeurs):
                l[nom] = bool(v)
    return lignes


def remplissage_case(binaire, mot):
    """Remplissage de la case à cocher située juste à gauche d'un mot : on cherche le contour carré le plus
    grand dans la zone, puis la part d'encre à l'intérieur (contour exclu). None si aucune case n'est vue."""
    x, y, _, h = mot["boite"]
    x0, y0 = max(0, x - int(2.5 * h)), max(0, y - 4)
    zone = (binaire[y0:y + h + 4, x0:max(x0 + 1, x - 1)] == 0).astype(np.uint8)
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    carres = [cv2.boundingRect(c) for c in contours]
    carres = [r for r in carres if 0.5 * h <= r[2] <= 2 * h and 0.5 * h <= r[3] <= 2 * h and 0.6 <= r[2] / r[3] <= 1.6]
    if not carres:
        return None
    cx, cy, cl, ch = max(carres, key=lambda r: r[2] * r[3])
    m = max(2, int(0.22 * min(cl, ch)))
    interieur = zone[cy + m:cy + ch - m, cx + m:cx + cl - m]
    return float(interieur.mean()) if interieur.size else None


def encre_avant(binaire, mot):
    """Part d'encre brute dans la zone à gauche d'un mot (repli quand la case n'est pas un carré net)."""
    x, y, _, h = mot["boite"]
    zone = binaire[max(0, y - 4):y + h + 4, max(0, x - int(2.5 * h)):max(1, x - 1)]
    return float((zone == 0).mean()) if zone.size else 0.0


def delai_coche(lignes_page, binaire):
    """Délai lu sur des cases à cocher, sur la ligne « Délai » : la case nettement la plus remplie l'emporte.
    Si une coche déborde de sa case (la case n'est plus un carré), on compare l'encre brute des zones."""
    ys = [l["boite"][1] for l in lignes_page if "delai" in sans_accents(l["texte"])]
    cases, brut, fusionnes = {}, {}, []
    for l in lignes_page:
        if not any(abs(l["boite"][1] - y) <= 30 for y in ys):
            continue
        for mot in l["mots"]:
            m = LIBELLE_DELAI.search(sans_accents(mot["texte"]))
            delai = {"nomal": "normal"}.get(m.group(1), m.group(1)) if m else None
            if delai and m.start() > 0:
                fusionnes.append(delai)            # coche lue comme quelques caractères collés au libellé
            if delai:
                brut[delai] = encre_avant(binaire, mot)
                r = remplissage_case(binaire, mot)
                if r is not None:
                    cases[delai] = r
    if len(fusionnes) == 1 and len(brut) >= 2:
        return dict(valeur=fusionnes[0], confiance=75.0,
                    ligne=f"coche collée au libellé « {fusionnes[0]} » (les autres cases sont vides)")
    if len(cases) < len(brut) and len(brut) >= 2:
        tri = sorted(brut.items(), key=lambda kv: -kv[1])
        if tri[0][1] >= 1.4 * tri[1][1]:
            return dict(valeur=tri[0][0], confiance=60.0,
                        ligne=f"encre (coche hors case) : {', '.join(f'{k} {v:.2f}' for k, v in tri)}")
    if not cases:
        return None
    tri = sorted(cases.items(), key=lambda kv: -kv[1])
    second = tri[1][1] if len(tri) > 1 else 0.0
    if tri[0][1] >= 0.12 and tri[0][1] >= 2 * second:
        return dict(valeur=tri[0][0], confiance=round(min(100.0, 100 * (tri[0][1] - second) / 0.25), 1),
                    ligne=f"cases : {', '.join(f'{k} {v:.2f}' for k, v in tri)}")
    return None


def alertes(doc):
    """Contrôles simples sur le résultat (préfigurent la brique 5)."""
    a = []
    for nom in ("numero_commande", "date_commande", "emetteur"):
        if not doc["champs"].get(nom):
            a.append(f"champ manquant : {nom}")
    sirets = list(doc["champs"].get("siret") or []) + [doc["champs"][k] for k in ("emetteur_siret",)
                                                       if (doc["champs"].get(k) or {}).get("cle_valide") is False]
    for s in {s["valeur"]: s for s in sirets}.values():
        if not s.get("cle_valide", True):
            a.append(f"SIRET {s['valeur']} : clé de Luhn invalide (erreur de lecture, ou SIRET fictif)")
    if not doc["echantillons"]:
        a.append("aucun tableau d'échantillons trouvé")
    elif doc["confiance_tableau"] is not None and doc["confiance_tableau"] < CONFIANCE_TABLEAU_MIN:
        a.append(f"tableau peu lisible (confiance OCR {doc['confiance_tableau']:.0f}) : à ressaisir ou vérifier")
    for i, e in enumerate(doc["echantillons"], 1):
        if any(c in e for c in COLONNES_COCHE) and not any(e.get(c) for c in COLONNES_COCHE) and e.get("reference"):
            a.append(f"échantillon {i} ({e.get('reference')}) : aucune analyse cochée")
    return a


def _lire_tesseract(pages):
    """Lecture Tesseract : (champs, lignes brutes du tableau, colonnes, confiance du tableau, texte par page)."""
    champs, rangees_toutes, colonnes, textes, conf_tableaux = {}, [], [], [], []
    for gris, binaire in pages:
        lignes = lignes_eparses(gris)
        textes.append("\n".join(l["texte"] for l in lignes))
        rangees, conf, noms, y_tableau = lire_tableau(gris, binaire)
        for nom, valeur in extraire_champs(lignes, y_tableau).items():
            if valeur and not champs.get(nom):
                champs[nom] = valeur
        if not champs.get("delai"):
            champs["delai"] = delai_coche(lignes, binaire)
        rangees_toutes += rangees
        colonnes = colonnes or noms
        if conf is not None:
            conf_tableaux.append(conf)
    return champs, rangees_toutes, colonnes, (round(float(np.mean(conf_tableaux)), 1) if conf_tableaux else None), textes


def _lire_vlm(pages):
    """Lecture par le modèle vision-langage local : (champs, lignes brutes du tableau, réponse complète)."""
    reponse = vlm.lire_bon([gris for gris, _ in pages])
    champs = vlm.champs_depuis_reponse(reponse)
    champs["laboratoire"] = recaler(champs.get("laboratoire"), REFERENTIEL["laboratoire"])
    siret = champs.get("emetteur_siret")
    if siret:
        chiffres = re.sub(r"\D", "", siret["valeur"])
        champs["emetteur_siret"] = {**siret, "valeur": chiffres, "cle_valide": len(chiffres) == 14 and luhn(chiffres)}
    return champs, vlm.echantillons_depuis_reponse(reponse), reponse


def extraire_document(pages, moteur="tesseract"):
    """pages : liste de (gris normalisé, binaire) dans l'ordre. moteur : « tesseract », « vlm » (modèle
    vision-langage local) ou « les deux » (lectures comparées, l'accord sert de confiance). Renvoie le dict du
    document : champs, échantillons reconstruits, colonnes, confiances, alertes, remarques, texte OCR."""
    alertes_moteurs, reponse_vlm = [], None
    if moteur in ("vlm", "les deux") and not vlm.disponible():
        raise RuntimeError(f"Modèle {vlm.MODELE} indisponible : lancer Ollama et « ollama pull {vlm.MODELE} »")
    if moteur in ("tesseract", "les deux"):
        champs_t, rangees_t, colonnes, conf_tableau, textes = _lire_tesseract(pages)
    if moteur == "tesseract":
        champs, rangees = champs_t, rangees_t
    else:
        champs_v, rangees_v, reponse_vlm = _lire_vlm(pages)
        rangees, colonnes = rangees_v, ["reference", "profondeur", "analyses", "remarques"]
        if moteur == "vlm":
            champs, conf_tableau, textes = champs_v, None, []
        else:
            champs, alertes_moteurs = fusionner_champs(champs_v, champs_t)
            if len(rangees_t) != len(rangees_v):
                alertes_moteurs.append(f"tableau : {len(rangees_v)} lignes lues par le modèle, {len(rangees_t)} par Tesseract")
    echantillons = reconstruire(rangees, (champs.get("numero_commande") or {}).get("valeur"),
                                (champs.get("emetteur") or {}).get("valeur"))
    doc = dict(moteur=moteur, champs=champs, echantillons=echantillons, colonnes=colonnes,
               confiance_tableau=conf_tableau, texte_ocr=textes, reponse_vlm=reponse_vlm)
    doc["alertes"] = list(dict.fromkeys(alertes(doc) + alertes_moteurs))
    doc["remarques"] = ([] if champs.get("client_final") else
                        ["pas de client final indiqué : les analyses sont pour l'émetteur lui-même"])
    annules = [e.get("reference", "?") for e in echantillons if e.get("statut") == "annule"]
    if annules:
        doc["remarques"].append(f"échantillon(s) annulé(s), à ne pas analyser : {', '.join(annules)}")
    return doc
