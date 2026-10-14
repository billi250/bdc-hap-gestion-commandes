"""Score de qualité d'une page prétraitée : est-elle prête pour l'OCR, ou faut-il la rescanner ?

Cinq critères, chacun ramené entre 0 (mauvais) et 1 (bon) par une rampe linéaire entre deux seuils :
- résolution : 150 dpi suffisent pour du texte de 10 points (inconnue pour une image seule : critère neutre) ;
- netteté : énergie des contours (variance du laplacien) rapportée au carré du contraste de l'encre, pour ne
  pas pénaliser deux fois un document pâle mais net ;
- contraste de l'encre : niveau de gris médian des pixels d'encre (plus il est sombre, mieux c'est) ;
- propreté : part des composantes de quelques pixels (bruit) parmi toutes les composantes du binaire ;
- orientation : confiance de la détection d'orientation.
Score = moyenne des critères x 100. Verdict « à rescanner » si un critère est nul ou si le score est sous 50,
« à vérifier » si un critère est sous 0,5 ou si le score est sous 80, sinon « prêt pour l'OCR ».
"""
import cv2
import numpy as np

RAMPES = dict(resolution=(100, 150), nettete=(0.01, 0.03), contraste=(215, 150), proprete=(0.6, 0.25),
              orientation=(0.1, 0.5))
NOMS = dict(resolution="résolution faible", nettete="image floue", contraste="encre trop pâle",
            proprete="bruit élevé", orientation="orientation incertaine")


def rampe(valeur, mauvais, bon):
    """Ramène une valeur entre 0 et 1 : 0 au seuil mauvais, 1 au seuil bon (les seuils peuvent être décroissants)."""
    if valeur is None:
        return None
    return float(np.clip((valeur - mauvais) / (bon - mauvais), 0.0, 1.0))


def mesures(gris_normalise, binaire, dpi, confiance_orientation):
    """Valeurs brutes des cinq critères."""
    encre = binaire == 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(encre.astype(np.uint8), connectivity=8)
    contraste = float(np.median(gris_normalise[encre])) if encre.any() else 255.0
    ecart = max(255.0 - contraste, 1.0)
    return dict(resolution=dpi, nettete=float(cv2.Laplacian(gris_normalise, cv2.CV_64F).var()) / ecart ** 2,
                contraste=contraste,
                proprete=float((stats[1:, cv2.CC_STAT_AREA] <= 6).sum() / max(n - 1, 1)),
                orientation=confiance_orientation)


def verdict(score, criteres):
    """Verdict lisible à partir du score et des critères (None = critère non mesurable)."""
    connus = [v for v in criteres.values() if v is not None]
    if score < 50 or min(connus) == 0:
        return "a rescanner"
    return "a verifier" if score < 80 or min(connus) < 0.5 else "pret pour OCR"


def evaluer_qualite(gris_normalise, binaire, dpi, confiance_orientation):
    """Renvoie un dict : score (0 à 100), verdict, critères (0 à 1), mesures brutes, raisons (critères < 0,5)."""
    brut = mesures(gris_normalise, binaire, dpi, confiance_orientation)
    criteres = {k: rampe(v, *RAMPES[k]) for k, v in brut.items()}
    connus = [v for v in criteres.values() if v is not None]
    score = round(100 * float(np.mean(connus)), 1)
    return dict(score=score, verdict=verdict(score, criteres),
                criteres={k: (round(v, 2) if v is not None else None) for k, v in criteres.items()},
                mesures={k: (round(v, 3) if isinstance(v, float) else v) for k, v in brut.items()},
                raisons=[NOMS[k] for k, v in criteres.items() if v is not None and v < 0.5])
