"""Détection des doublons : même document reçu deux fois (renvoi, deuxième scan, « annule et remplace »).

Deux étapes, calées sur le corpus :
1. Empreinte perceptuelle (DCT d'une vignette 32 x 32, 64 bits) : une faible distance de Hamming signale deux
   pages du même modèle de formulaire. Elle ne suffit pas : deux bons différents d'un même émetteur ont souvent
   la même empreinte, car le formulaire imprimé domine l'image.
2. Pour ces paires, on aligne finement les deux pages (points ORB et homographie RANSAC), puis on mesure la
   part de l'encre d'une page absente de l'autre (à 3 pixels près). Sur le corpus : 0,1 à 0,9 % pour un même
   document rescanné (simulé), au moins 2,1 % pour deux documents différents du même formulaire (valeurs
   manuscrites différentes). Un vrai re-scan rebinarisé d'un carbone pâle monte à 2,5 % : les deux plages se
   recouvrent. D'où trois niveaux : « quasi identique » (<= 1,5 %, aucun faux positif sur le corpus),
   « a verifier » (<= 4 % : ressemblance forte, même document ou même formulaire peu rempli), « meme modele ».
   Pour confirmer un doublon, il faudra comparer les champs extraits (numéro de commande) en brique 3.
"""
import cv2
import numpy as np

SEUIL_MODELE = 10          # bits différents (sur 64) : au plus, même modèle (12 au minimum entre émetteurs différents)
SEUIL_CONTENU = 0.015      # part d'encre non partagée en dessous de laquelle les pages sont quasi identiques
SEUIL_PROCHE = 0.04        # en dessous : ressemblance forte, à vérifier à la main
COTE_ALIGNEMENT = 1000


def empreinte(binaire):
    """Empreinte perceptuelle 64 bits (chaîne hexadécimale de 16 caractères) d'une page."""
    petit = cv2.resize(binaire, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    basses = cv2.dct(petit)[:8, :8].ravel()
    bits = (basses > np.median(basses[1:])).astype(int)
    bits[0] = 0
    return "".join(f"{int(''.join(map(str, bits[i:i + 4])), 2):x}" for i in range(0, 64, 4))


def distance(e1, e2):
    """Nombre de bits différents entre deux empreintes."""
    return bin(int(e1, 16) ^ int(e2, 16)).count("1")


def _reduire(binaire):
    f = COTE_ALIGNEMENT / max(binaire.shape)
    return cv2.resize(binaire, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def aligner(reference, autre):
    """Aligne autre sur reference (points ORB, homographie RANSAC). Renvoie l'image alignée ou None."""
    orb = cv2.ORB_create(4000, fastThreshold=10)
    k_ref, d_ref = orb.detectAndCompute(reference, None)
    k_aut, d_aut = orb.detectAndCompute(autre, None)
    if d_ref is None or d_aut is None:
        return None
    paires = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d_aut, d_ref), key=lambda m: m.distance)[:800]
    if len(paires) < 30:
        return None
    h, _ = cv2.findHomography(np.float32([k_aut[m.queryIdx].pt for m in paires]),
                              np.float32([k_ref[m.trainIdx].pt for m in paires]), cv2.RANSAC, 3.0)
    if h is None:
        return None
    return cv2.warpPerspective(autre, h, (reference.shape[1], reference.shape[0]), borderValue=255)


def ecart_contenu(binaire_a, binaire_b):
    """Part de l'encre d'une page absente de l'autre après alignement (la plus grande des deux), ou None
    si l'alignement échoue."""
    a, b = _reduire(binaire_a), _reduire(binaire_b)
    b = aligner(a, b)
    if b is None:
        return None
    ea, eb = (a < 128).astype(np.uint8), (b < 128).astype(np.uint8)
    noyau = np.ones((7, 7), np.uint8)
    hors_a = (eb & (1 - cv2.dilate(ea, noyau))).sum() / max(int(eb.sum()), 1)
    hors_b = (ea & (1 - cv2.dilate(eb, noyau))).sum() / max(int(ea.sum()), 1)
    return float(max(hors_a, hors_b))


def comparer(binaire_a, binaire_b, empreinte_a=None, empreinte_b=None):
    """Renvoie (niveau, distance d'empreinte, écart de contenu). niveau : « quasi identique », « a verifier »,
    « meme modele » ou None (pages différentes)."""
    d = distance(empreinte_a or empreinte(binaire_a), empreinte_b or empreinte(binaire_b))
    if d > SEUIL_MODELE:
        return None, d, None
    e = ecart_contenu(binaire_a, binaire_b)
    if e is None or e > SEUIL_PROCHE:
        return "meme modele", d, e
    return ("quasi identique" if e <= SEUIL_CONTENU else "a verifier"), d, e


def paires_proches(pages):
    """pages : {identifiant: image binaire}. Renvoie les paires ressemblantes [(id1, id2, niveau, distance,
    écart)], les quasi identiques en premier."""
    empreintes = {k: empreinte(v) for k, v in pages.items()}
    ids = sorted(pages)
    paires = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            niveau, d, e = comparer(pages[a], pages[b], empreintes[a], empreintes[b])
            if niveau:
                paires.append((a, b, niveau, d, None if e is None else round(e, 4)))
    ordre = {"quasi identique": 0, "a verifier": 1, "meme modele": 2}
    return sorted(paires, key=lambda p: (ordre[p[2]], p[4] if p[4] is not None else 1))
