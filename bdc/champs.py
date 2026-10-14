"""Extraction des champs d'en-tête à partir des lignes OCR : règles et expressions régulières tolérantes.

Chaque champ est rendu sous la forme dict(valeur, confiance 0-100, ligne source). Les contrôles simples
(clé de Luhn du SIRET, date plausible) abaissent la confiance et produisent des alertes.
"""
import re
import unicodedata
from difflib import SequenceMatcher

DATE = re.compile(r"\b(\d{1,2})\s?[/.-]\s?(\d{1,2})\s?[/.-]\s?(\d{4}|\d{2})\b")
MOIS = ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre",
        "novembre", "decembre"]
DATE_LETTRES = re.compile(rf"\b(\d{{1,2}})(?:er)?\s+({'|'.join(MOIS)})\s+(\d{{4}})\b")
TELEPHONE = re.compile(r"(?<!\d)(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+\s?@\s?[\w-]+(?:\.[\w-]+)+")
SIRET = re.compile(r"(?<!\d)((?:\d[\s.]?){13}\d)(?!\d)")
NUMEROS = [  # du plus spécifique au plus général
    re.compile(r"N\s?[°ºo]\s*[:.]?\s*([A-Z]{1,5}\s?[-/]\s?\d{2,4}\s?[-/]\s?\d{2,5}(?:[-/]\d+)?)(?![/\d])"),
    re.compile(r"R[ée]f\.?\s*[:.]?\s*([A-Z]{1,5}-\d{2,4}-\d{2,5})"),
    re.compile(r"N\s?[°ºo]\s*[:.]?\s*(\d{4,8})\b"),
]
# Étiquettes par rôle, par ordre de préférence. Les étiquettes de plusieurs mots sont assez spécifiques pour
# se passer du « : » (souvent perdu par l'OCR) ; les étiquettes d'un seul mot l'exigent.
ETIQUETTES = {
    "contact_emetteur": [r"charg[ée] d.affaires", r"technicien", r"responsable"],
    "client_final": [r"client final(?:\s*/\s*site)?", r"raison sociale", r"exploitant", r"demandeur\s*[:.]"],
    "client_interlocuteur": [r"interlocuteur\s*[:.]?", r"contact\s*[:.]"],
    "adresse_site": [r"adresse du site", r"adresse\s*[:.]"],
    "parcelles": [r"lieu-dit\s*/\s*parcelles", r"parcelle\s*\(?s\)?", r"lieu-dit\s*[:.]", r"parcelles\s*[:.]"],
    "commune": [r"commune\s*[:.]"],
    "laboratoire": [r"^\W*laboratoire\s*[:.]?", r"destinataire\s*[:.]"],
}
TOUTES = "|".join(e for liste in ETIQUETTES.values() for e in liste) + r"|\bT[ée]l\b|E-?mail|SIRET|Date\b"
TITRE = re.compile(r"\b(BON|DEMANDE|COMMANDE|FICHE)\b.*$")
DELAIS = ("standard", "normal", "urgent", "express")
REFERENTIEL = {"laboratoire": ["ORVENNE ANALYSES"]}   # valeurs connues (référentiel du LIMS)
LIBELLE_DELAI = re.compile(r"(standard|normal|nomal|urgent|express)")


def sans_accents(texte):
    """Minuscules sans accents, pour comparer des libellés lus par l'OCR."""
    return "".join(c for c in unicodedata.normalize("NFD", texte.lower()) if unicodedata.category(c) != "Mn")


def luhn(chiffres):
    """Vrai si la suite de chiffres respecte la clé de Luhn (contrôle des numéros SIREN / SIRET)."""
    total = 0
    for i, c in enumerate(reversed(chiffres)):
        n = int(c) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def _champ(valeur, ligne, confiance=None):
    return dict(valeur=valeur.strip(" :;.,-"), confiance=round(ligne["confiance"] if confiance is None else confiance, 1),
                ligne=ligne["texte"])


def emetteur(lignes):
    """Première ligne lisible du haut de la page (en-tête de l'émetteur), sans le titre du document qui
    partage souvent la même ligne (« ... BON DE COMMANDE »)."""
    for l in lignes:
        if sum(c.isdigit() for c in l["texte"]) > 0.25 * len(l["texte"]):
            continue                                            # en-tête de transmission d'un fax
        nom = TITRE.sub("", l["texte"]).strip(" |-")
        nom = re.sub(r"^\W*\w?\s+(?=[A-Z]{3})", "", nom)        # débris d'un logo lu comme une lettre
        if len(re.sub(r"\W", "", nom)) >= 4 and l["confiance"] >= 50:
            return _champ(nom, l)
    return None


def numero_commande(lignes):
    """Numéro de commande ou référence du bon (ex : MST/2026/118, VB-26-0471, 004587)."""
    for motif in NUMEROS:
        for l in lignes:
            m = motif.search(l["texte"])
            if m:
                return _champ(re.sub(r"\s+", "", m.group(1)), l)
    return None


def _date_valide(m):
    j, mo, a = int(m.group(1)), int(m.group(2)), m.group(3)
    return 1 <= j <= 31 and 1 <= mo <= 12, f"{j:02d}/{mo:02d}/{a if len(a) == 4 else '20' + a}"


def _premiere_date(lignes):
    """Première date valide (jj/mm/aaaa ou « 26 septembre 2026 ») des lignes données."""
    for l in lignes:
        for m in DATE.finditer(l["texte"]):
            ok, date = _date_valide(m)
            if ok:
                return _champ(date, l)
        m = DATE_LETTRES.search(sans_accents(l["texte"]))
        if m and 1 <= int(m.group(1)) <= 31:
            return _champ(f"{int(m.group(1)):02d}/{MOIS.index(m.group(2)) + 1:02d}/{m.group(3)}", l)
    return None


def date_commande(lignes, y_tableau=None):
    """Date du bon, par ordre de préférence : « date de commande » ou « émise le », ligne du numéro de
    commande, « Date : », « Fait à ..., le » ou « ville, le ... ». Les dates de prélèvement et les dates lues
    ailleurs (tableau) sont écartées : mieux vaut un champ manquant signalé qu'une date fausse."""
    t = [sans_accents(l["texte"]) for l in lignes]
    hors_prelevement = [(l, s) for l, s in zip(lignes, t) if not re.search(r"prel[eè]v", s)]
    niveaux = [r"date de commande|emise le|date d.emission", r"n\s?[°o]\s*[:.]?\s*[a-z0-9]", r"\bdate\s*[:.]",
               r"fait a .*\ble\b|,\s*le \d"]
    for motif in niveaux:
        trouve = _premiere_date([l for l, s in hors_prelevement if re.search(motif, s)])
        if trouve:
            return trouve
    if y_tableau is not None:      # étiquette « Date » mal lue : une date au-dessus du tableau, suivie de « : »
        return _premiere_date([l for l, s in hors_prelevement if l["boite"][1] < y_tableau and ":" in s])
    return None


def date_prelevement(lignes):
    """Date de prélèvement (première date d'une ligne qui parle de prélèvement)."""
    return _premiere_date([l for l in lignes if re.search(r"prel[eè]v", sans_accents(l["texte"]))])


def par_etiquette(lignes, nom):
    """Valeur écrite après une étiquette du rôle demandé (ex : « Client final / site : SCEA Clos des Aubiers »),
    coupée à l'étiquette suivante de la même ligne (« ... Chargé d'affaires : ... », « ... Tél. : ... »).
    Une valeur avec moins de trois lettres (numéro seul) est écartée."""
    for etiquette in ETIQUETTES[nom]:
        motif = re.compile(rf"(?:{etiquette})\s*[:.]?\s*(.+)", re.IGNORECASE)
        for l in lignes:
            m = motif.search(l["texte"])
            if not m:
                continue
            valeur = re.split(rf"\s(?:{TOUTES})|\s{{2,}}|\s+[-–]\s+|\s(?=0\d[\s.]?\d{{2}}\b)", m.group(1),
                              flags=re.IGNORECASE)[0]
            if len(re.findall(r"[A-Za-zÀ-ÿ]", valeur)) >= 3:
                return _champ(valeur, l)
    return None


def sirets(lignes):
    """Numéros SIRET (14 chiffres) sur les lignes qui les annoncent, sans doublon, avec le résultat du
    contrôle de la clé de Luhn (un SIRET réel la respecte ; les SIRET fictifs du corpus, souvent non)."""
    trouves = {}
    for l in lignes:
        if not re.search(r"siret|n.? ?siret|siren", sans_accents(l["texte"])):
            continue
        for m in SIRET.finditer(l["texte"]):
            chiffres = re.sub(r"\D", "", m.group(1))
            trouves.setdefault(chiffres, {**_champ(chiffres, l), "cle_valide": luhn(chiffres)})
    return list(trouves.values())


def contacts(lignes):
    """Téléphones (format 0X XX XX XX XX) et adresses électroniques, sans doublon."""
    tels, mails = {}, {}
    for l in lignes:
        for m in TELEPHONE.finditer(l["texte"]):
            chiffres = re.sub(r"\D", "", m.group(0))
            numero = "0" + chiffres[2:] if chiffres.startswith("33") else chiffres
            tels.setdefault(numero, _champ(" ".join(numero[i:i + 2] for i in range(0, 10, 2)), l))
        for m in EMAIL.finditer(l["texte"]):
            adresse = m.group(0).replace(" ", "")
            mails.setdefault(adresse.lower(), _champ(adresse, l))
    return list(tels.values()), list(mails.values())


def delai_ecrit(lignes):
    """Délai écrit en toutes lettres (« Délai : standard (10 jours ouvrés) ») ; les cases à cocher sont lues
    à part (voir extraction.delai_coche)."""
    for l in lignes:
        t = sans_accents(l["texte"])
        cites = {d for d in DELAIS if d in t}
        if "delai" in t and len(cites) == 1 and "[" not in l["texte"] and "□" not in l["texte"]:
            return _champ(cites.pop(), l)      # un seul délai cité : il est écrit, pas coché
    return None


def zone_entete(lignes):
    """Ordonnée de fin de l'en-tête de l'émetteur : première ligne qui porte une étiquette de formulaire
    (date, client, chargé d'affaires...). Ce qui est au-dessus appartient à l'émetteur."""
    for l in lignes:
        if re.search(rf"{TOUTES}|date\s*[:.]", l["texte"], re.IGNORECASE) and not re.search(r"SIRET|T[ée]l", l["texte"]):
            return l["boite"][1]
    return None


def par_role(champs, lignes, y_entete):
    """Répartit téléphones et SIRET entre l'émetteur (en-tête) et le client (lignes du formulaire)."""
    dans_entete = lambda c: y_entete is None or any(l["texte"] == c["ligne"] and l["boite"][1] < y_entete for l in lignes)
    roles = {}
    for nom in ("telephones", "siret"):
        for c in champs[nom]:
            roles.setdefault(f"{'emetteur' if dans_entete(c) else 'client'}_{nom.rstrip('s')}", c)
    return roles


def extraire_champs(lignes, y_tableau=None):
    """Champs d'en-tête d'une page, rangés par rôle : émetteur (qui envoie le bon, avec son adresse, son SIRET,
    son téléphone et son chargé d'affaires), client final (pour qui sont faites les analyses, avec site,
    parcelles, téléphone, interlocuteur), laboratoire destinataire, commande (numéro, dates, délai)."""
    tels, mails = contacts(lignes)
    em = emetteur(lignes)
    y_entete = zone_entete(lignes)
    entete = [l for l in lignes if y_entete is None or l["boite"][1] < y_entete]
    adresse_em = next((_champ(l["texte"], l) for l in entete if re.search(r"\b\d{5}\b", l["texte"])
                       and not re.search(r"SIRET|T[ée]l|\d{2}\s\d{2}\s\d{2}", l["texte"])), None)
    champs = dict(emetteur=em, emetteur_adresse=adresse_em,
                  numero_commande=numero_commande(lignes), date_commande=date_commande(lignes, y_tableau),
                  date_prelevement=date_prelevement(lignes), delai=delai_ecrit(lignes),
                  laboratoire=par_etiquette(lignes, "laboratoire"), emails=mails, telephones=tels,
                  siret=sirets(lignes))
    for role in ("contact_emetteur", "client_final", "client_interlocuteur", "adresse_site", "parcelles", "commune"):
        champs[role] = par_etiquette(lignes, role)
    champs.update(par_role(champs, lignes, y_entete))
    for nom, connues in REFERENTIEL.items():
        champs[nom] = recaler(champs.get(nom), connues)
    return champs


def recaler(champ, connues, seuil=0.6):
    """Remplace une valeur lue par la valeur connue la plus proche du référentiel, si elle est assez proche
    (ex : « Onenne Analycec » -> « ORVENNE ANALYSES ») ; la valeur lue est gardée dans « lue »."""
    if not champ:
        return champ
    meilleure = max(connues, key=lambda v: SequenceMatcher(None, sans_accents(champ["valeur"]), sans_accents(v)).ratio())
    if SequenceMatcher(None, sans_accents(champ["valeur"]), sans_accents(meilleure)).ratio() >= seuil:
        return {**champ, "valeur": meilleure, "lue": champ["valeur"]}
    return champ
