"""Lecture OCR locale (Tesseract, français) des pages prétraitées.

L'OCR se fait sur l'image en niveaux de gris normalisée (plus fiable que le binaire sur le corpus), agrandie
deux fois : Tesseract lit mieux vers 300 dpi que les 150 dpi des scans.
"""
import cv2
import numpy as np
import pytesseract

LANGUE = "fra"
AGRANDISSEMENT = 2


def disponible():
    """Vrai si Tesseract et le français sont installés."""
    try:
        return LANGUE in pytesseract.get_languages(config="")
    except (pytesseract.TesseractNotFoundError, OSError):
        return False


def _agrandir(image, facteur=AGRANDISSEMENT):
    return cv2.resize(image, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_CUBIC)


def lignes_ocr(gris):
    """Lit la page et renvoie ses lignes : liste de dict (texte, confiance moyenne 0-100, boîte x, y, l, h
    dans les coordonnées de l'image d'origine, mots avec leur boîte et leur confiance)."""
    donnees = pytesseract.image_to_data(_agrandir(gris), lang=LANGUE, config="--psm 3",
                                        output_type=pytesseract.Output.DICT)
    lignes = {}
    for i, mot in enumerate(donnees["text"]):
        if not mot.strip() or float(donnees["conf"][i]) < 0:
            continue
        cle = (donnees["block_num"][i], donnees["par_num"][i], donnees["line_num"][i])
        boite = [v // AGRANDISSEMENT for v in (donnees["left"][i], donnees["top"][i], donnees["width"][i], donnees["height"][i])]
        lignes.setdefault(cle, []).append(dict(texte=mot, confiance=float(donnees["conf"][i]), boite=boite))
    sortie = []
    for mots in lignes.values():
        xs = [m["boite"][0] for m in mots] + [m["boite"][0] + m["boite"][2] for m in mots]
        ys = [m["boite"][1] for m in mots] + [m["boite"][1] + m["boite"][3] for m in mots]
        sortie.append(dict(texte=" ".join(m["texte"] for m in mots),
                           confiance=round(float(np.mean([m["confiance"] for m in mots])), 1),
                           boite=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], mots=mots))
    return sorted(sortie, key=lambda l: (l["boite"][1], l["boite"][0]))


def lire_cellule(gris, rect, plusieurs_lignes=False):
    """Texte et confiance d'une cellule de tableau (agrandissement x3). Une ligne de texte est attendue,
    sauf pour les en-têtes de colonnes, souvent sur deux lignes (plusieurs_lignes=True)."""
    x0, y0, x1, y1 = rect
    if x1 - x0 < 5 or y1 - y0 < 5:
        return "", 0.0
    zone = cv2.copyMakeBorder(gris[y0:y1, x0:x1], 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    d = pytesseract.image_to_data(_agrandir(zone, 3), lang=LANGUE, config="--psm 6" if plusieurs_lignes else "--psm 7",
                                  output_type=pytesseract.Output.DICT)
    mots = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    if not mots:
        return "", 0.0
    return " ".join(t for t, _ in mots), round(float(np.mean([c for _, c in mots])), 1)


def lignes_eparses(gris):
    """Lecture mot à mot (mode « texte épars », --psm 11), puis regroupement des mots en lignes par position.
    Sur un formulaire, les petites étiquettes imprimées voisines d'une grande écriture manuscrite y sont bien
    mieux lues qu'en lecture de page (--psm 3). Deux mots très proches sont recollés (« 30/09/20 » + « 26 »)."""
    d = pytesseract.image_to_data(_agrandir(gris), lang=LANGUE, config="--psm 11", output_type=pytesseract.Output.DICT)
    mots = [dict(texte=d["text"][i], confiance=float(d["conf"][i]),
                 boite=[v // AGRANDISSEMENT for v in (d["left"][i], d["top"][i], d["width"][i], d["height"][i])])
            for i in range(len(d["text"])) if d["text"][i].strip() and float(d["conf"][i]) >= 0]
    lignes = []
    for m in sorted(mots, key=lambda m: m["boite"][1] + m["boite"][3] / 2):
        centre = m["boite"][1] + m["boite"][3] / 2
        proches = [l for l in lignes if abs(l["centre"] - centre) <= 0.6 * min(l["hauteur"], m["boite"][3]) + 2]
        if proches:
            l = min(proches, key=lambda l: abs(l["centre"] - centre))
            l["mots"].append(m)
            l["centre"] = float(np.median([w["boite"][1] + w["boite"][3] / 2 for w in l["mots"]]))
            l["hauteur"] = float(np.median([w["boite"][3] for w in l["mots"]]))
        else:
            lignes.append(dict(centre=centre, hauteur=m["boite"][3], mots=[m]))
    sortie = []
    for l in lignes:
        mots_l = sorted(l["mots"], key=lambda m: m["boite"][0])
        texte = mots_l[0]["texte"]
        for avant, m in zip(mots_l, mots_l[1:]):
            ecart = m["boite"][0] - (avant["boite"][0] + avant["boite"][2])
            texte += ("" if ecart < 0.25 * max(avant["boite"][3], m["boite"][3]) and avant["texte"][-1].isdigit()
                      and m["texte"][0].isdigit() else " ") + m["texte"]
        xs = [m["boite"][0] for m in mots_l] + [m["boite"][0] + m["boite"][2] for m in mots_l]
        ys = [m["boite"][1] for m in mots_l] + [m["boite"][1] + m["boite"][3] for m in mots_l]
        sortie.append(dict(texte=texte, confiance=round(float(np.mean([m["confiance"] for m in mots_l])), 1),
                           boite=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], mots=mots_l))
    return sorted(sortie, key=lambda l: l["boite"][1])
