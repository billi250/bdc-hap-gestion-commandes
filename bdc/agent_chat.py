"""Assistant des laborantins : un agent qui répond aux questions sur les BDC en appelant des fonctions.

Trois règles :
- il interroge la base, il ne devine pas : chaque réponse s'appuie sur ce que renvoient les fonctions ;
- il cite sa source : le n° interne du BDC (lien vers le scan dans l'interface) et le n° de commande ;
- il ne modifie rien : toutes les fonctions sont en lecture seule ; les corrections passent par l'écran
  de revue.

Boucle : le modèle choisit une ou plusieurs fonctions, on les exécute, on lui rend les résultats, et ainsi
de suite jusqu'à sa réponse (au plus ETAPES_MAX allers-retours). Chaque appel est tracé et affiché.
"""
import json

from . import controles, gcms, stockage
from . import config as cfg

ETAPES_MAX = 6
RESULTATS_MAX = 25

SYSTEME = """Tu es l'assistant des laborantins d'un laboratoire d'analyses de sols. Tu réponds en français
courant, de façon brève et directe, sans tirets longs. Tu ne connais les bons de commande (BDC) que par les
fonctions fournies : appelle-les avant de répondre, ne devine jamais. Cite toujours tes sources sous la forme
« BDC n° <id> (commande <numéro>) ». Si les fonctions ne renvoient rien, dis-le simplement. Tu ne modifies
rien : pour corriger un BDC, renvoie le laborantin vers l'écran de revue. Statuts possibles : brouillon,
a verifier, valide. Date du jour : {aujourdhui}."""

PROPRIETES_RECHERCHE = {
    "texte": "texte cherché partout (n°, client, référence d'échantillon, analyse)",
    "numero": "numéro de commande (ou une partie)", "client": "nom de l'émetteur ou du client final (ou une partie)",
    "date_min": "date de commande minimale JJ/MM/AAAA", "date_max": "date de commande maximale JJ/MM/AAAA",
    "statut": "brouillon, a verifier ou valide", "analyse": "code d'analyse, ex. HAP 16",
    "echantillon": "référence d'échantillon (ou une partie)"}

OUTILS = [
    dict(name="chercher_bdc", description="Cherche des BDC selon des critères (tous facultatifs, combinés).",
         parameters_json_schema={"type": "object", "properties": {
             k: ({"type": "string", "enum": ["brouillon", "a verifier", "valide"], "description": d} if k == "statut"
                 else {"type": "string", "description": d}) for k, d in PROPRIETES_RECHERCHE.items()}}),
    dict(name="details_bdc", description="Contenu complet d'un BDC : champs, échantillons, anomalies, statut.",
         parameters_json_schema={"type": "object", "properties": {"id": {"type": "integer",
                                 "description": "n° interne du BDC"}}, "required": ["id"]}),
    dict(name="statistiques", description="Nombre de BDC reçus, validés, en attente, délai moyen de validation "
                                          "par client.", parameters_json_schema={"type": "object", "properties": {}}),
    dict(name="alertes_delai", description="BDC urgents ou en attente depuis trop longtemps.",
         parameters_json_schema={"type": "object", "properties": {}}),
    dict(name="file_a_verifier", description="BDC non validés, du plus douteux (score bas) au plus sûr.",
         parameters_json_schema={"type": "object", "properties": {}}),
    dict(name="resultats_gcms", description="Résultats HAP (GC-MS) d'un échantillon, si disponibles.",
         parameters_json_schema={"type": "object", "properties": {"reference": {"type": "string",
                                 "description": "référence de l'échantillon"}}, "required": ["reference"]}),
]


class Outils:
    """Fonctions en lecture seule offertes à l'agent. Chaque appel note les BDC cités (sources)."""

    def __init__(self, con, seuils, chemin_gcms=None):
        self.con, self.seuils, self.chemin_gcms = con, seuils, chemin_gcms
        self.sources = []

    def _citer(self, lignes):
        for l in lignes:
            if l.get("id") not in self.sources:
                self.sources.append(l["id"])
        return lignes

    def chercher_bdc(self, **criteres):
        lignes = stockage.chercher(self.con, **{k: v for k, v in criteres.items() if v})
        resume = [{k: l[k] for k in ("id", "numero", "emetteur", "client", "date_commande", "delai", "statut",
                                     "nb_echantillons", "score")} for l in lignes[:RESULTATS_MAX]]
        return dict(nombre=len(lignes), bdc=self._citer(resume))

    def details_bdc(self, id):
        d = stockage.lire(self.con, int(id))
        if d is None:
            return dict(erreur=f"aucun BDC n° {id}")
        self._citer([d])
        b = d["courant"]
        champs = {n: (b.get(n) or {}).get("valeur") for n in b if isinstance(b.get(n), dict)}
        ech = [dict(reference=e.get("reference"), profondeur=e.get("profondeur"), analyses=e.get("analyses"),
                    annule=controles.annule(e), observations=e.get("observations")) for e in b.get("echantillons") or []]
        return dict(id=d["id"], fichier=d["fichier"], statut=d["statut"], recu_le=d["recu_le"],
                    valide_le=d["valide_le"], valide_par=d["valide_par"], champs=champs, echantillons=ech,
                    anomalies=[a["message"] for a in d["anomalies"] or []],
                    corrections=len(stockage.historique(self.con, d["id"])))

    def statistiques(self):
        return stockage.statistiques(self.con)

    def alertes_delai(self):
        return dict(alertes=self._citer(controles.alertes_delai(stockage.lister(self.con), self.seuils)))

    def file_a_verifier(self):
        lignes = [l for l in stockage.lister(self.con) if l["statut"] != "valide"]
        lignes.sort(key=lambda l: l["score"] if l["score"] is not None else 0)
        return dict(file=self._citer([{k: l[k] for k in ("id", "numero", "emetteur", "statut", "score")}
                                      for l in lignes[:RESULTATS_MAX]]))

    def resultats_gcms(self, reference):
        return gcms.resultats(reference, self.chemin_gcms)

    def executer(self, nom, arguments):
        """Exécute une fonction par son nom ; une erreur est rendue au modèle au lieu d'arrêter l'agent."""
        if nom not in {o["name"] for o in OUTILS}:
            return dict(erreur=f"fonction inconnue : {nom}")
        try:
            return getattr(self, nom)(**(arguments or {}))
        except Exception as e:
            return dict(erreur=str(e))


def repondre(question, passerelle, outils, historique=(), aujourdhui=""):
    """Pose la question à l'agent. historique : [(rôle « user » ou « model », texte)] des échanges précédents.
    Renvoie dict(reponse, appels [(fonction, arguments, résultat)], sources [id de BDC])."""
    if passerelle.locale:
        return dict(reponse="L'assistant n'est pas disponible en variante locale : il utilise l'appel de "
                            "fonctions du modèle par API.", appels=[], sources=[])
    from google.genai import types
    contenus = [types.Content(role=r, parts=[types.Part(text=t)]) for r, t in historique]
    contenus.append(types.Content(role="user", parts=[types.Part(text=question)]))
    declarations = [types.FunctionDeclaration(**o) for o in OUTILS]
    appels = []
    for _ in range(ETAPES_MAX):
        reponse, _ = passerelle.gemini("chatbot", contenus, justification="question d'un laborantin (texte des BDC)",
                                       outils=declarations, consigne_systeme=SYSTEME.format(aujourdhui=aujourdhui))
        demandes = reponse.function_calls or []
        if not demandes:
            return dict(reponse=(reponse.text or "").strip(), appels=appels, sources=outils.sources)
        contenus.append(reponse.candidates[0].content)      # garde les signatures de réflexion du modèle
        parties = []
        for d in demandes:
            resultat = outils.executer(d.name, dict(d.args or {}))
            appels.append((d.name, dict(d.args or {}), resultat))
            parties.append(types.Part.from_function_response(
                name=d.name, response={"resultat": json.loads(json.dumps(resultat, default=str))}))
        contenus.append(types.Content(role="user", parts=parties))
    return dict(reponse="Je n'ai pas abouti en assez peu d'étapes : reformulez la question plus précisément.",
                appels=appels, sources=outils.sources)


def outils_par_defaut(con):
    """Outils branchés sur la base et les seuils de config.yaml."""
    return Outils(con, cfg.charger()["seuils"], cfg.chemin("gcms_resultats"))
