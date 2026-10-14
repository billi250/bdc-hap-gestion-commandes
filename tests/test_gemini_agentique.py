"""Tests de la chaîne par modèle API : schéma, passerelle tracée, contrôles, base, export, agents.
Aucun appel réseau : le client Gemini est remplacé par des réponses préparées."""
import json
from datetime import date, datetime, timedelta

import cv2
import pytest
from google.genai import errors

from bdc import agent_chat, agent_relecture, controles, export, extraction_llm, stockage, verite
from bdc import config as cfg
from bdc.modeles import ModeleIndisponible, Passerelle, lire_envois
from bdc.schema import BonDeCommande, Relecture
from bdc.traitement import traiter

from faux_gemini import FauxClient, appel, config_test, texte

REF = {"laboratoires": ["ORVENNE ANALYSES"], "clients": [{"nom": "ECOTELLUS", "code_lims": "CLI-006"}],
       "analyses": [{"code": "HAP 16", "code_lims": "HAP16", "synonymes": ["HAP16"]},
                    {"code": "MS", "code_lims": "MS", "synonymes": ["matière sèche"]}]}
SEUILS = cfg.charger()["seuils"]
AUJOURDHUI = date(2026, 10, 13)


def lecture(**autres):
    """Réponse type du modèle pour un BDC (dict au format du schéma)."""
    c = lambda v, conf=95, boite=None: dict(valeur=v, confiance=conf, page=1, boite=boite or [100, 100, 130, 400])
    base = dict(numero_commande=c("004587"), date_commande=c("30/09/2026"), delai=c("Urgent"),
                emetteur_nom=c("ECOTELLUS SAS"), emetteur_siret=c("732 829 320 00074"),
                laboratoire=c("ORVENNE ANALYSES"), client_nom=c("SCEA Clos des Aubiers"),
                echantillons=[dict(reference=f"ECO-4587-{i}", profondeur="0-30", analyses=["HAP16", "matière sèche"],
                                   ligne_barree=i == 4, confiance=90, page=1, boite=[500 + 30 * i, 50, 525 + 30 * i, 950])
                              for i in range(1, 6)])
    base.update(autres)
    return base


def bdc_dict(**autres):
    b = BonDeCommande.model_validate(lecture(**autres))
    extraction_llm.recaler(b, REF)
    return b.model_dump()


# --- schéma --------------------------------------------------------------------------------------------
def test_schema_normalise_delai_et_boites():
    b = BonDeCommande.model_validate(dict(delai=dict(valeur="Urgent J+5", confiance=90),
                                          numero_commande=dict(valeur="  ", boite=[50, 10, 20, 30])))
    assert b.delai.valeur == "urgent"
    assert b.numero_commande.valeur is None and b.numero_commande.boite is None


def test_proche_nom_suivi_d_un_complement():
    assert extraction_llm.proche("ECOTELLUS ENVIRONNEMENT", ["ECOTELLUS", "AGRISOL CONSEIL"]) == "ECOTELLUS"
    assert extraction_llm.proche("ECO", ["ECOTELLUS"]) is None


def test_recaler_referentiel_et_codes_analyse():
    b = BonDeCommande.model_validate(lecture())
    recalages = extraction_llm.recaler(b, REF)
    assert b.emetteur_nom.valeur == "ECOTELLUS"
    assert b.emetteur_siret.valeur == "73282932000074"
    assert b.echantillons[0].analyses == ["HAP 16", "MS"]
    assert any(r["champ"] == "emetteur_nom" and r["lu"] == "ECOTELLUS SAS" for r in recalages)


# --- passerelle ------------------------------------------------------------------------------------------
def test_passerelle_trace_avant_envoi_et_valide_le_schema(tmp_path):
    journal = tmp_path / "envois.jsonl"
    client = FauxClient([texte(lecture())])
    p = Passerelle(client=client, journal=journal)
    image = cv2.imencode(".png", 255 * __import__("numpy").ones((20, 20), "uint8"))[1].tobytes()
    bdc, infos = p.generer_json("extraction", "lis", [image], BonDeCommande, document="doc.pdf", justification="test")
    assert bdc.numero_commande.valeur == "004587"
    envois = lire_envois(journal)
    assert [e["statut"] for e in envois] == ["envoi", "reçu"]
    assert envois[0]["document"] == "doc.pdf" and len(envois[0]["images_sha256"][0]) == 64
    assert client.requetes[0]["model"] == cfg.modele("extraction")["modele"]


def test_passerelle_bascule_sur_le_secours(tmp_path):
    panne = errors.ServerError(503, {"error": {"message": "surcharge"}})
    client = FauxClient([panne, texte(lecture())])
    p = Passerelle(client=client, journal=tmp_path / "e.jsonl", attentes=())
    p.generer_json("extraction", "lis", [], BonDeCommande)
    assert [r["model"] for r in client.requetes] == [cfg.modele("extraction")["modele"], cfg.modele("secours")["modele"]]


def test_passerelle_reessaie_apres_surcharge(tmp_path):
    panne = errors.ServerError(503, {"error": {"message": "surcharge"}})
    client = FauxClient([panne, texte(lecture())])
    Passerelle(client=client, journal=tmp_path / "e.jsonl", attentes=(0,)).generer_json("extraction", "lis", [], BonDeCommande)
    assert [r["model"] for r in client.requetes] == [cfg.modele("extraction")["modele"]] * 2


def test_passerelle_reponse_hors_schema_detectee(tmp_path):
    p = Passerelle(client=FauxClient([texte({"numero_commande": {"confiance": 300}})]), journal=tmp_path / "e.jsonl")
    with pytest.raises(Exception):
        p.generer_json("extraction", "lis", [], BonDeCommande)


def test_sans_cle_modele_indisponible(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "cle_api", lambda: None)
    with pytest.raises(ModeleIndisponible):
        Passerelle(journal=tmp_path / "e.jsonl").client()


# --- contrôles -------------------------------------------------------------------------------------------
def test_controles_bdc_correct_sans_erreur():
    a = controles.controler(bdc_dict(), REF, SEUILS, aujourd_hui=AUJOURDHUI)
    assert not [x for x in a if x["niveau"] == "erreur"]


def test_controles_detectent_les_incoherences():
    b = bdc_dict(date_commande=dict(valeur="31/02/2026", confiance=90), numero_commande=dict(valeur=None),
                 nb_echantillons_annonce=dict(valeur="6", confiance=90), emetteur_siret=dict(valeur="12345678901234"))
    messages = " | ".join(x["message"] for x in controles.controler(b, REF, SEUILS, aujourd_hui=AUJOURDHUI))
    assert "date impossible" in messages and "N° de commande" in messages
    assert "6 échantillons annoncés, 4 listés" in messages and "Luhn" in messages


def test_controles_doublon_meme_numero_meme_emetteur():
    autres = [dict(id=1, fichier="fax.pdf", sha256="x", numero="004587", emetteur="ECOTELLUS")]
    a = controles.controle_doublons(bdc_dict(), "y", autres)
    assert a and "reçu deux fois" in a[0]["message"]


def test_alertes_delai():
    il_y_a = lambda j: (datetime.now().astimezone() - timedelta(days=j)).isoformat()
    lignes = [dict(id=1, statut="a verifier", delai="urgent", recu_le=il_y_a(2), numero="1", emetteur="A"),
              dict(id=2, statut="brouillon", delai="standard", recu_le=il_y_a(5), numero="2", emetteur="B"),
              dict(id=3, statut="valide", delai="urgent", recu_le=il_y_a(9), numero="3", emetteur="C"),
              dict(id=4, statut="brouillon", delai="standard", recu_le=il_y_a(0), numero="4", emetteur="D")]
    assert [a["id"] for a in controles.alertes_delai(lignes, SEUILS)] == [2, 1]


# --- base et export --------------------------------------------------------------------------------------
def test_base_corrections_tracees_et_export_des_seuls_valides(tmp_path):
    con = stockage.ouvrir(tmp_path / "bdc.sqlite")
    b = bdc_dict()
    i = stockage.enregistrer(con, "a.pdf", b, b, [], 90, "brouillon", {})
    j = stockage.enregistrer(con, "b.pdf", b, b, [], 90, "brouillon", {})
    corrige = json.loads(json.dumps(b))
    corrige["client_telephone"]["valeur"] = "06 39 98 20 64"
    assert stockage.corriger(con, i, corrige, "Marie") == 1
    stockage.changer_statut(con, i, "valide", "Marie")
    hist = stockage.historique(con, i)
    assert hist[0]["champ"] == "client_telephone" and hist[0]["qui"] == "Marie" and hist[-1]["apres"] == "valide"
    lignes = export.lignes_lims([stockage.lire(con, i), stockage.lire(con, j)], REF)
    assert {l["n_bdc_interne"] for l in lignes} == {i}
    assert len(lignes) == 4 * 2                      # 4 échantillons non annulés x 2 analyses
    assert lignes[0]["code_client_lims"] == "CLI-006" and lignes[0]["code_analyse_lims"] == "HAP16"
    assert "ECO-4587-4" not in {l["reference_echantillon"] for l in lignes}
    assert export.csv_lims([stockage.lire(con, i)], REF).startswith("﻿".encode())
    corrige["delai"]["valeur"] = "standard"
    stockage.corriger(con, i, corrige, "Paul")
    assert stockage.lire(con, i)["statut"] == "a verifier"      # modifié après validation : à revalider


def test_recherche_et_statistiques(tmp_path):
    con = stockage.ouvrir(tmp_path / "bdc.sqlite")
    b = bdc_dict()
    stockage.enregistrer(con, "a.pdf", b, b, [], 90, "a verifier", {})
    assert len(stockage.chercher(con, client="ecotel", echantillon="4587-3", analyse="HAP")) == 1
    assert stockage.chercher(con, date_min="01/10/2026") == []
    assert stockage.statistiques(con)["par_emetteur"]["ECOTELLUS"]["en_attente"] == 1


# --- agents ----------------------------------------------------------------------------------------------
class FaussePasserelle:
    locale = False

    def __init__(self, relectures):
        self.relectures = list(relectures)
        self.demandes = []

    def reglage(self, usage):
        return dict(fournisseur="test", modele="faux")

    def generer_json(self, usage, consigne, images, schema, document="", justification=""):
        self.demandes.append(consigne)
        return Relecture(**self.relectures.pop(0)), {}


def test_agent_relecture_corrige_le_champ_en_erreur():
    b = bdc_dict(date_commande=dict(valeur="30/09/2O26", confiance=60, page=1, boite=[100, 600, 130, 800]))
    page = 255 * __import__("numpy").ones((1400, 1000), "uint8")
    p = FaussePasserelle([dict(valeur="30/09/2026", confiance=92, explication="O lu comme 0")])
    anomalies, journal = agent_relecture.relire(b, [page], p, REF, SEUILS,
                                                controle=lambda x: controles.controler(x, REF, SEUILS, aujourd_hui=AUJOURDHUI))
    assert b["date_commande"]["valeur"] == "30/09/2026"
    assert "Date de commande" in p.demandes[0] and "date impossible" in p.demandes[0]
    action = next(e for e in journal if e["type"] == "action")
    assert action["retenu"] and action["avant"] == "30/09/2O26"
    assert not [a for a in anomalies if a["champ"] == "date_commande"]


def test_agent_relecture_ne_degrade_pas():
    b = bdc_dict(emetteur_telephone=dict(valeur="05 56 12 34 56", confiance=50, page=1, boite=[10, 10, 40, 300]))
    page = 255 * __import__("numpy").ones((1400, 1000), "uint8")
    p = FaussePasserelle([dict(valeur="05 56 12 34 5", confiance=40)])
    agent_relecture.relire(b, [page], p, REF, SEUILS, controle=lambda x: controles.controler(x, REF, SEUILS, aujourd_hui=AUJOURDHUI))
    assert b["emetteur_telephone"]["valeur"] == "05 56 12 34 56"


def test_agent_chat_appelle_la_base_et_cite_ses_sources(tmp_path):
    con = stockage.ouvrir(tmp_path / "bdc.sqlite")
    b = bdc_dict()
    i = stockage.enregistrer(con, "a.pdf", b, b, [], 90, "a verifier", {})
    client = FauxClient([appel("chercher_bdc", client="ECOTELLUS", statut="a verifier"),
                         texte(f"Un BDC en attente : BDC n° {i} (commande 004587).")])
    outils = agent_chat.Outils(con, SEUILS)
    r = agent_chat.repondre("Quels BDC d'ECOTELLUS sont en attente ?", Passerelle(client=client, journal=tmp_path / "e.jsonl"), outils)
    assert r["appels"][0][0] == "chercher_bdc" and r["appels"][0][2]["nombre"] == 1
    assert r["sources"] == [i] and "004587" in r["reponse"]
    tours = client.requetes[1]["contents"]
    assert tours[-1].parts[0].function_response.name == "chercher_bdc"


def test_agent_chat_fonction_inconnue_sans_plantage(tmp_path):
    outils = agent_chat.Outils(stockage.ouvrir(tmp_path / "b.sqlite"), SEUILS)
    assert "erreur" in outils.executer("supprimer_bdc", {"id": 1})


# --- vérité et chaîne complète ---------------------------------------------------------------------------
def test_verite_noter():
    b = bdc_dict()
    v = dict(partielle=True, champs=dict(numero_commande="4587", client_telephone=None, delai="urgent"),
             echantillons=[dict(reference=f"ECO-4587-{k}", annule=k == 4) for k in range(1, 6)])
    n = verite.noter(b, v)
    assert n["justes"] == n["total"] == 2 + 10 + 1


def test_chaine_complete_avec_faux_modele(tmp_path, page):
    conf = config_test(tmp_path)
    con = stockage.ouvrir(conf["chemins"]["base"])
    octets = cv2.imencode(".png", page)[1].tobytes()
    client = FauxClient([texte(lecture())] * 2)
    p = Passerelle(conf, client=client, journal=conf["chemins"]["envois_externes"])
    etapes = []
    i = traiter(octets, "essai.png", p, con, REF, conf, etapes.append)
    d = stockage.lire(con, i)
    assert d["numero"] == "004587" and d["emetteur"] == "ECOTELLUS" and d["nb_echantillons"] == 4
    assert d["extrait"]["emetteur_nom"]["valeur"] == "ECOTELLUS"
    assert any("Enregistré" in e["message"] for e in etapes)
    j = traiter(octets, "essai_bis.png", p, con, REF, conf)
    assert any(a["champ"] == "doublon" for a in stockage.lire(con, j)["anomalies"])
