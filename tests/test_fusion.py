"""Fusion des lectures Qwen3-VL / Tesseract (sans appeler le modèle) et lecture tolérante de sa réponse."""
from bdc import vlm
from bdc.fusion import accord, fusionner_champs


def champ(v, c=None):
    return dict(valeur=v, confiance=c, ligne="test")


def test_numeriques_identiques_seulement():
    assert accord("06 39 98 20 64", "0639982064", numerique=True) == 1.0
    assert accord("06 39 18 20 64", "06 34 18 20 64", numerique=True) == 0.0


def test_regles_de_fusion():
    vl = {"client_telephone": champ("06 39 18 20 64"), "emetteur_adresse": champ("21 avenue des Plantes"),
          "contact_emetteur": champ("T. RENAUDIN"), "emetteur": champ("ECOTELLUS ENVIRONNEMENT"), "commune": None}
    te = {"client_telephone": champ("06 34 18 20 64", 89), "emetteur_adresse": champ("21 avenue des Platanes", 94),
          "contact_emetteur": champ("T. Re naudin", 81), "emetteur": champ("ECOTELLUS", 94), "commune": None}
    f, alertes = fusionner_champs(vl, te)
    assert f["client_telephone"]["confiance"] == 55 and any("client_telephone" in a for a in alertes)
    assert f["emetteur_adresse"]["valeur"] == "21 avenue des Platanes" and f["emetteur_adresse"]["source"] == "tesseract"
    assert f["contact_emetteur"]["valeur"] == "T. RENAUDIN" and f["contact_emetteur"]["confiance"] == 95
    assert f["emetteur"]["confiance"] == 80 and not any(a.startswith("emetteur :") for a in alertes)
    assert f["commune"] is None


def test_reponse_du_modele_cles_tolerees():
    brut = {"emetteur": {"nom": "ECOTELLUS", "SIRET": "794"}, "client_final": {"raison_sociale": "SCEA"},
            "dates": {"date_commande": "30/09/2026"}, "délai": "Urgent J+5",
            "tableau_échantillons": [{"réf": "ECO-4587-4", "analyses": ["HAP 16"], "ligne_barree": True}]}
    r = vlm._cles(brut)
    r["delai"] = vlm._delai(r.get("delai"))
    ch = vlm.champs_depuis_reponse(r)
    assert ch["client_final"]["valeur"] == "SCEA" and ch["date_commande"]["valeur"] == "30/09/2026"
    assert ch["delai"]["valeur"] == "urgent" and ch["emetteur_siret"]["valeur"] == "794"
    assert vlm.echantillons_depuis_reponse(r)[0] == dict(reference="ECO-4587-4", profondeur="", analyses="HAP 16",
                                                         remarques="", barree=True)
