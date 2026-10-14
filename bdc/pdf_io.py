"""Lecture des documents : PDF (une image par page, ou page rendue) et images (PNG, JPEG, TIFF...)."""
from pathlib import Path

import cv2
import numpy as np
import pymupdf

PT_PAR_POUCE = 72
DPI_RENDU = 150          # résolution de rendu quand la page n'est pas une simple image (PDF généré, texte vectoriel)
EXTENSIONS_IMAGE = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def lister_pdf(dossier):
    """Renvoie la liste triée des fichiers PDF d'un dossier."""
    return sorted(Path(dossier).glob("*.pdf"))


def _decoder(octets):
    """Décode une image JPEG/PNG en tableau numpy, sans changer sa profondeur de couleur."""
    return cv2.imdecode(np.frombuffer(octets, np.uint8), cv2.IMREAD_UNCHANGED)


def _image_page(doc, page):
    """Image d'une page. Si la page est un scan (une seule image qui la couvre), on extrait l'image native,
    sans rééchantillonnage ; sinon (PDF généré, plusieurs images, texte), on rend la page à DPI_RENDU.
    Renvoie (image, mode)."""
    images = page.get_images(full=True)
    if len(images) == 1:
        cadres = page.get_image_rects(images[0][0])
        if cadres and cadres[0].width >= 0.9 * page.rect.width and cadres[0].height >= 0.9 * page.rect.height:
            image = _decoder(doc.extract_image(images[0][0])["image"])
            if image is not None:
                return image, "image native"
    pix = page.get_pixmap(dpi=DPI_RENDU, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy(), f"rendu {DPI_RENDU} dpi"


def _pages_document(doc):
    """Pages d'un document PyMuPDF ouvert : liste de dict (page, image, dpi_x, dpi_y, largeur_pt, hauteur_pt, mode)."""
    pages = []
    for i, page in enumerate(doc, start=1):
        image, mode = _image_page(doc, page)
        l_pt, h_pt = page.rect.width, page.rect.height
        pages.append(dict(page=i, image=image, largeur_pt=l_pt, hauteur_pt=h_pt, mode=mode,
                          dpi_x=image.shape[1] / (l_pt / PT_PAR_POUCE),
                          dpi_y=image.shape[0] / (h_pt / PT_PAR_POUCE)))
    return pages


def pages_pdf(chemin):
    """Extrait l'image de chaque page d'un PDF (voir _image_page). La résolution est déduite de la taille
    de l'image rapportée à la taille de la page en points (72 points par pouce)."""
    with pymupdf.open(chemin) as doc:
        return _pages_document(doc)


def pages_depuis_octets(octets, nom):
    """Pages d'un document reçu en mémoire (téléversement) : PDF ou image. Pour une image, la résolution
    n'est pas connue (dpi à None) ; pour un TIFF multipage, seule la première page est lue."""
    extension = Path(nom).suffix.lower()
    if extension == ".pdf":
        with pymupdf.open(stream=octets, filetype="pdf") as doc:
            return _pages_document(doc)
    if extension in EXTENSIONS_IMAGE:
        image = _decoder(octets)
        if image is None:
            raise ValueError(f"Image illisible : {nom}")
        return [dict(page=1, image=image, largeur_pt=None, hauteur_pt=None, mode="image", dpi_x=None, dpi_y=None)]
    raise ValueError(f"Format non pris en charge : {extension} (PDF ou image attendus)")


def en_gris(image):
    """Convertit une image (gris, BGR ou BGRA, 8 ou 16 bits) en niveaux de gris 8 bits."""
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image, alpha=255.0 / max(float(image.max()), 1.0))
    if image.ndim == 2:
        return image
    code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code)
