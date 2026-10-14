# Lecture de bons de commande HAP

Projet fil rouge HETIC. Briques 0 (corpus) et 1 (prétraitement), lecture des BDC par modèle à vision
(Gemini par API, ou Qwen3-VL local), contrôles, agents et interface des laborantins (voir plus bas).

Corpus : 28 PDF dans `data/BDC/` (7 émetteurs x 4 documents, 32 pages : BDC_05 fait 2 pages). Chaque page est
une image JPEG sans texte, à 150 ou 170 dpi. Les données sont fictives mais traitées comme réelles (RGPD) :
`data/`, `out/`, les PDF et les zip ne sont jamais versionnés (voir `.gitignore`). Tout le traitement est local.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Testé avec Python 3.13. Versions figées dans `requirements.txt` (OpenCV, numpy, PyMuPDF, Streamlit, pytesseract,
pytest). L'extraction (brique 3) demande aussi Tesseract avec le français, en local :
`brew install tesseract tesseract-lang` (macOS). Sans Tesseract, les briques 0 et 1 fonctionnent normalement.

## Commande unique

```bash
python run_all.py data/BDC out/
```

Elle enchaîne l'inventaire, le prétraitement et la planche contact, puis affiche un résumé (pages traitées,
pages dont l'orientation ou la perspective n'a pas pu être corrigée avec confiance). Environ 16 s pour 32 pages.

| Sortie | Contenu |
|---|---|
| `out/inventory.csv` | une ligne par page : fichier, nb de pages, taille en pixels, dpi, format en mm, canaux, taille, SHA-256 |
| `out/pages/<pdf>_p<n>.png` | image native de chaque page, sans rééchantillonnage |
| `out/preprocessed/<pdf>_p<n>.png` | page prétraitée, binaire (encre noire sur blanc) |
| `out/preprocessed/<pdf>_p<n>_gris.png` | page prétraitée en niveaux de gris normalisés (utile pour l'OCR) |
| `out/preprocess_log.jsonl` | journal : un enregistrement JSON par page (voir plus bas) |
| `out/contact_sheet.png` | page 1 de chaque PDF, avant et après, côte à côte |

Les briques se lancent aussi séparément : `python inventory.py data/BDC out/`, `python preprocess.py data/BDC out/`.
Mesure de l'orientation contre la vérité établie à l'oeil : `python evaluer.py out/preprocess_log.jsonl verite_terrain.json`.
Tests : `python -m pytest`.

Garde-fous : le scan d'origine n'est jamais modifié (lecture seule, empreinte SHA-256 dans l'inventaire et le
journal, test dédié) et le programme refuse d'écrire dans le dossier des données ou un de ses sous-dossiers.

## Interface pour un nouveau document

```bash
streamlit run app.py --server.address localhost
```

Puis ouvrir http://localhost:8501. On dépose un ou plusieurs documents (PDF ou image PNG, JPEG, TIFF...) ; chaque
page est prétraitée et affichée avant / après, avec l'orientation, le redressement, la perspective, les alertes et
le journal JSON. Les réglages de normalisation sont modifiables dans la barre latérale. Un bouton télécharge les
pages prétraitées et le journal (zip). Si `data/BDC/` existe, on peut aussi choisir un exemple du corpus.

Le document est traité en mémoire sur le poste : rien n'est envoyé à l'extérieur ni écrit sur le disque.
`--server.address localhost` empêche l'accès depuis le réseau. Un PDF scanné est lu à sa résolution native ;
un PDF généré par logiciel (texte vectoriel, plusieurs images) est rendu à 150 dpi.

## Organisation

| Fichier | Responsabilité |
|---|---|
| `bdc/pdf_io.py` | lecture des PDF, extraction de l'image native et de la résolution |
| `bdc/encre.py` | masque d'encre réduit, retrait des longs traits |
| `bdc/orientation.py` | orientation 0 / 90 / 180 / 270 |
| `bdc/redressement.py` | inclinaison résiduelle |
| `bdc/perspective.py` | détection de photo, 4 coins, transformation de perspective |
| `bdc/normalisation.py` | débruitage, retrait du fond, binarisation, nettoyage |
| `bdc/journal.py` | journal JSONL |
| `bdc/fichiers.py` | empreinte, écriture PNG, garde-fou sur le dossier des données |
| `bdc/planche.py` | planche contact |
| `inventory.py`, `preprocess.py`, `run_all.py`, `evaluer.py` | points d'entrée en ligne de commande |
| `app.py` | interface Streamlit pour un nouveau document |
| `bdc/ocr.py`, `bdc/champs.py`, `bdc/tableau.py`, `bdc/extraction.py`, `extract.py` | extraction (brique 3) |
| `bdc/diagnostic.py`, `bdc/qualite.py`, `bdc/doublons.py` | diagnostic du défaut, score de qualité, doublons |

## Choix techniques

Ordre du prétraitement : **perspective, axe des lignes, redressement, haut/bas, normalisation**. La perspective
passe d'abord car l'orientation se mesure mieux sur une feuille remise à plat ; le redressement passe avant la
décision haut/bas car une inclinaison de 1 à 3° fausse cette décision, alors que le redressement ne dépend pas du sens.

**Orientation** (pas de Tesseract sur le poste, et pas d'envoi à un service externe) : heuristique sans OCR, à la
résolution native.
1. Axe : on retire les longs traits (cadres, tableaux), puis une fermeture horizontale relie les caractères d'une
   ligne en blocs larges et peu hauts. On garde l'axe (0 ou 90°) qui donne la plus grande longueur cumulée de blocs.
2. Haut/bas : en écriture latine, les jambages montants (b, d, l, t, majuscules, accents) sont plus fréquents que
   les descendants (g, p, q, y). Chaque ligne vote selon l'encre au-dessus et au-dessous du coeur de la ligne.

Mesure sur les 32 pages (vérité établie à l'oeil, `verite_terrain.json`) : **32/32 justes**, confiance minimale
0,80. Une première version fondée sur le profil de toute la page n'en trouvait que 18/32 (trompée par les codes-barres,
le bruit de fax et les tableaux), une seconde 26/32 tant que le redressement venait après.

**Redressement** : profil de projection. On fait tourner le masque d'encre de -5 à +5° (pas de 0,25° puis 0,02°) et
on garde l'angle qui maximise les variations du profil des lignes (lignes de texte et traits de tableau bien à plat).
Plus robuste que Hough sur des pages mêlant texte, tableaux et manuscrit. Mesures : BDC_06_n1 -2,49° (2,5° annoncé),
BDC_07_n1 -1,30° (1,3° annoncé) ; inclinaison résiduelle après traitement : 0,06° en moyenne, 0,07° au maximum.

**Perspective** : une page est considérée comme une photo si sa bordure est nettement plus sombre que son centre.
On cherche alors le plus grand contour clair (seuil d'Otsu), son enveloppe convexe, puis une approximation à
4 sommets. Si les 4 coins ne sont pas trouvés, l'image n'est pas déformée et l'échec est journalisé. Les côtés mesurés
étant raccourcis par la perspective, le rapport largeur/hauteur est ramené au format A4 quand il en est proche
(journalisé « A4 impose »). Résultat : les 4 photos BDC_03 corrigées, aucun scan pris à tort pour une photo.

**Normalisation** : débruitage par moyennes non locales ; atténuation des ombres et des taches par division par le
fond estimé (dilatation puis médiane, qui effacent l'encre) ; binarisation adaptative gaussienne (bloc 31, constante 15) ;
remplissage des grands aplats noirs que le seuil local évide (bandeaux de titre) ; suppression des points de bruit
isolés en protégeant les deux-points (deux points alignés verticalement) ; suppression des traces de bord
(photocopie). Tous les paramètres sont écrits dans le journal.

**Journal** (`out/preprocess_log.jsonl`) : fichier source et son SHA-256, page, angle d'orientation et confiance,
scores intermédiaires, angle de redressement, perspective (statut, coins, format), paramètres de normalisation,
tailles d'entrée et de sortie, alertes, durée.

## Extraction (brique 3, première version)

```bash
python extract.py out/
```

À lancer après `run_all.py`. Écrit `out/extraction/<pdf>.json`, `champs.csv` (une ligne par document) et
`echantillons.csv` (une ligne par échantillon). Environ 4 s par document. Dans l'interface, l'extraction s'active
d'un interrupteur ; les champs et le tableau s'affichent dans des **tableaux modifiables** (la personne corrige ce
que l'OCR a mal lu) et l'extraction corrigée se télécharge en JSON.

Méthode : OCR Tesseract (français, local) sur l'image grise normalisée agrandie deux fois ; champs d'en-tête par
règles et expressions régulières tolérantes (émetteur, numéro de commande, date de commande, date de prélèvement,
client, adresse, site, interlocuteur, SIRET, téléphones, e-mails) ; délai lu sur les cases à cocher (encre à
l'intérieur de la case, à gauche du libellé) ou en toutes lettres ; tableau des échantillons repéré par sa grille
(traits horizontaux et verticaux), chaque cellule lue à part, les colonnes d'analyses (HAP 16, MS, pH, ETM, COT)
lues comme des cases marquées ou non. Chaque champ porte une confiance (OCR) et la ligne d'où il vient. Contrôles :
champs manquants, clé de Luhn du SIRET, tableau peu lisible, échantillon sans analyse cochée.

Mesures sur le corpus (28 documents) : émetteur et client 28/28, tableau trouvé 28/28 (143 échantillons),
téléphone 26, délai 22, SIRET 21, numéro de commande 19, date de prélèvement 19, date de commande 15, e-mail 15.
Contrôlé à l'oeil sur les 7 documents n1 : numéros justes là où ils existent (5/5), tableaux imprimés lus presque
sans faute (BDC_01, 02, 05 : confiance 84 à 94), cases du délai justes sur BDC_01 à 03.

Rôles distingués : **émetteur** (qui envoie le bon : nom, adresse, SIRET, téléphone, chargé d'affaires),
**client final** (pour qui sont faites les analyses : nom, adresse du site, parcelles, commune, téléphone,
interlocuteur), **laboratoire** destinataire (recalé sur un petit référentiel : « Onenne Analycec » devient
ORVENNE ANALYSES, la valeur lue est gardée) et **commande** (numéro, dates, délai). Les téléphones et SIRET sont
attribués à l'émetteur s'ils sont dans l'en-tête, au client s'ils sont sur les lignes du formulaire. Les champs
sont lus en mode « texte épars » (Tesseract --psm 11), qui lit bien mieux les petites étiquettes imprimées
voisines d'une grande écriture manuscrite.

Tableau reconstruit (`bdc/reconstruction.py`) : une **ligne barrée** (long trait horizontal à mi-hauteur) passe au
statut « annule », confirmé si les observations disent « annulé » ; les **références** sont recalées sur le modèle
du bon (vote caractère par caractère, recoupé avec le numéro de commande et le nom de l'émetteur), les profondeurs
et les analyses (HAP 16, MS, pH, ETM, COT) normalisées, les références citées dans les observations recalées.
Chaque correction est tracée. Exemple BDC_06_n1 : « ELO-4597-1, €CO0-4587-2, EC0-4587-3, LÉ, EC0-87-5 » devient
ECO-4587-1 à 5, la ligne 4 barrée est annulée (« annulé (flacon cassé) »), « 9-30 » devient « 0-30 ». Délai
« Urgent » retrouvé même quand la coche, débordant de sa case, est lue collée au libellé (« Wurgent »).

**Deuxième moteur : modèle vision-langage local (Qwen3-VL 8B, via Ollama).** Tesseract est fait pour l'imprimé
et lit mal le manuscrit ; un modèle vision-langage lit la page entière, manuscrit compris, et rend les champs
structurés. Il tourne sur le poste (Mac M1 Pro, 16 Go : environ 30 s par page), aucune image ne sort.
Installation : `ollama pull qwen3-vl:8b`. Usage : `python extract.py out/ --moteur "les deux" --document BDC_06`,
ou le sélecteur « Moteur de lecture » de l'interface. En mode « les deux », les lectures sont comparées
(`bdc/fusion.py`) : chiffres identiques exigés pour les numéros, valeur de Tesseract gardée sur l'imprimé lu
sûrement, confiance 95 / 80 / 55 selon l'accord, champ signalé quand les lectures diffèrent.

Comparaison sur BDC_06_n1 (vérité lue sur le bon) :

| Champ | Vérité | Tesseract | Qwen3-VL | Retenu (les deux) |
|---|---|---|---|---|
| Chargé d'affaires | T. Renaudin | T. Re naudin | T. RENAUDIN | T. RENAUDIN (95) |
| Client final | SCEA Clos des Aubiers (vignoble) | SCEA Co dec Aubire | SCEA Cos des Aubiers | SCEA Cos des Aubiers (80) |
| Adresse du site | Route des Graves, 33580 St-Vernac | Route des Grave 33580 ct. Vernac | exact | exact (80) |
| Parcelles | AK 77 / AK 78 | exact | AK 27 / AK 78 | signalé (55) |
| Téléphone client | 06 39 98 20 64 | 06 34 18 20 64 | 06 39 18 20 64 | signalé (55) |
| Adresse émetteur | 21 avenue des Platanes | exact | 21 avenue des Plantes | exact, Tesseract (80) |
| Tableau | 5 lignes, ligne 4 barrée | références mal lues, recalées | exact, ligne 4 barrée | exact, ECO-4587-4 annulé |

Aucun moteur n'est infaillible sur le manuscrit : l'intérêt des deux est que leurs désaccords désignent les
champs à vérifier.

Limites de cette version : écriture manuscrite mal lue par Tesseract (BDC_03, BDC_06 : valeurs approximatives),
fax très dégradé (BDC_04 : tableau illisible, signalé), petit texte gris non lu (date de BDC_07), coches qui
débordent de leur case (BDC_06), en-têtes de colonnes parfois mal lus. Les SIRET du corpus échouent tous au
contrôle de Luhn : ils sont fictifs (le contrôle reste utile sur des données réelles).

## Ajouts au-delà du cahier des charges

**Diagnostic du défaut** (`bdc/diagnostic.py`) : sur l'image d'origine, chaque page est classée photo, fax, papier
teinté (carbone), photocopie, encre pâle, tache ou ombre, ou propre, d'après des signatures simples (bordure sombre,
image quasi binaire, fond non blanc, absence de noir franc, points isolés, creux du fond). Résultat conforme aux
7 émetteurs sur les 32 pages. Le défaut choisit les réglages de normalisation : par exemple, une constante de
binarisation abaissée de 15 à 10 pour l'encre pâle rend lisible le manuscrit de BDC_02 (« 8 éch. reçus, glacière »),
presque effacé avec les réglages communs. Désactivable (réglages manuels) dans l'interface.

**Score de qualité** (`bdc/qualite.py`) : note de 0 à 100 et verdict « prêt pour l'OCR », « à vérifier » ou
« à rescanner », à partir de la résolution, de la netteté (rapportée au contraste), du contraste de l'encre, du
bruit résiduel et de la confiance d'orientation, avec les raisons. Corpus : 25 pages prêtes, 7 à vérifier (fax
bruités, photocopies pâles, une page un peu floue). Il prépare la brique 5 (contrôles et confiance).

**Doublons** (`bdc/doublons.py`, `out/doublons.csv`) : empreinte perceptuelle pour trouver les pages du même
formulaire, puis alignement fin (points ORB, homographie) et part d'encre non partagée. Trois niveaux :
« quasi identique » (<= 1,5 %), « à vérifier » (<= 4 %), « même modèle ». Corpus : aucun doublon, 8 ressemblances
fortes, toutes entre bons d'un même émetteur ; un re-scan simulé de BDC_03 ou BDC_04 est reconnu comme quasi identique.
Dans l'interface, un nouveau document est comparé aux documents déjà traités (`out/preprocessed`).

## Limites connues

- Orientation : heuristique fondée sur l'écriture latine et validée sur 32 pages seulement. Une page presque sans
  texte, ou entièrement manuscrite, risque une décision incertaine (signalée par l'alerte « orientation incertaine »).
- Fax (BDC_04) : il reste des points de bruit de 4 à 6 pixels. Ils ont la même taille que les points des deux-points
  et des virgules, que l'on préfère conserver pour l'extraction des champs.
- Le manuscrit pâle et les tampons gris clair (BDC_02, BDC_06) sont partiellement fragmentés par la binarisation :
  l'image `_gris` normalisée est fournie pour la brique 4 (ratures, manuscrit, tampons).
- Le filigrane « COPIE » de BDC_07 disparaît à la binarisation (souhaitable pour l'OCR, mais l'information
  « copie » n'est plus visible sur l'image binaire).
- Perspective : la méthode suppose une feuille claire sur un fond plus sombre ; une feuille photographiée sur fond
  blanc ne serait pas détectée comme photo. Le recalage au format A4 suppose des bons de commande au format A4.
- Redressement limité à +/- 5° ; au-delà, seule l'orientation par quarts de tour s'applique.
- Diagnostic : l'anneau de café de BDC_02, fin et pâle, n'est pas détecté comme tache (BDC_02 est classé « encre
  pâle », ce qu'il est aussi). Les seuils sont calés sur 28 documents seulement.
- Doublons : la comparaison d'images ne distingue pas toujours un même document rescanné (jusqu'à 2,5 % d'écart
  pour un carbone pâle) d'un autre bon du même formulaire peu rempli (dès 2,1 %) : ces cas sortent « à vérifier ».
  La confirmation passera par les champs extraits (numéro de commande) en brique 3.

## Lecture par Gemini, agents et interface des laborantins

Décision (document « BDC : benchmark OCR et LLM ») : un modèle appelé par API, dont l'identifiant figé est
écrit dans un seul fichier, `config.yaml`. Choix retenu : **Gemini, niveau gratuit** (`gemini-3.1-flash-lite`, secours
`gemini-3.5-flash-lite`, choisis par le benchmark du 13/10/2026 : `gemini-3.8-flash` était saturé), identifiants stables relevés sur https://ai.google.dev/gemini-api/docs/models le
13/10/2026. Jamais d'alias `-latest` ni de modèle preview. Changer de version = changer une ligne, puis
relancer le benchmark.

### Mise en route

```bash
pip install -r requirements.txt
cp .env.exemple .env          # puis coller la clé Gemini après GEMINI_API_KEY=
streamlit run app.py --server.address localhost
```

La clé n'est jamais dans le code : elle est lue dans `GEMINI_API_KEY` (variable d'environnement ou `.env`,
non versionné). Le vrai référentiel (`referentiel.yaml` : clients, codes LIMS) n'est pas versionné non plus ;
`referentiel.exemple.yaml` sert de modèle.

### Pages de l'interface

| Page | Rôle |
|---|---|
| Nouveau BDC | dépôt d'un PDF ou d'une image ; prétraitement, lecture par le modèle, contrôles, agent de relecture ; le BDC entre en base |
| À vérifier | file des BDC non validés, les plus douteux d'abord ; alertes délai (urgent, attente trop longue) |
| Revue | scan à gauche, champs à droite (modifiables) ; champs douteux en orange ; « voir » surligne la zone sur le scan ; ajout d'échantillons et d'analyses depuis le référentiel ; statut brouillon / à vérifier / validé ; historique des corrections ; résultats GC-MS des échantillons |
| Recherche | par n° de commande, client, date, échantillon, analyse, statut, ou texte libre |
| Assistant | questions en français courant ; l'agent appelle des fonctions de recherche (lecture seule) et cite le n° de BDC |
| Tableau de bord | BDC reçus, validés, en attente, délai moyen de validation par client ; journal des envois externes |
| Export LIMS | CSV (une ligne par analyse, codes LIMS) et JSON des seuls BDC validés |
| Prétraitement (local) | l'ancienne interface : prétraitement et lecture Tesseract / Qwen3-VL, sans envoi |

### Agents

- **Agent de relecture** (`bdc/agent_relecture.py`) : après la lecture, les contrôles listent les anomalies ;
  l'agent choisit les champs à relire (erreurs d'abord), découpe et agrandit leur zone sur la page, demande au
  modèle de relire ce seul champ en lui donnant la règle qui a échoué, garde la nouvelle valeur seulement si
  elle est plus sûre et ne crée pas d'anomalie, puis recontrôle (2 tours, 6 champs au plus). Chaque étape est
  affichée et chaque changement est tracé comme une correction d'origine « agent ».
- **Assistant** (`bdc/agent_chat.py`) : appel de fonctions Gemini (`chercher_bdc`, `details_bdc`,
  `statistiques`, `alertes_delai`, `file_a_verifier`, `resultats_gcms`). Il interroge la base, cite ses
  sources, ne modifie rien.

### Garde-fous

- Le scan d'origine n'est jamais modifié : le document déposé est copié dans `out/scans/`, la lecture du modèle
  est figée (« extrait ») et les corrections sont stockées à côté (`out/bdc.sqlite`).
- Chaque envoi à Google est tracé avant l'envoi dans `out/envois_externes.jsonl` (date, document, modèle,
  empreinte SHA-256 des images, justification). Seules les pages prétraitées, ou un découpage d'une zone, sont
  envoyées. Variante locale (interrupteur dans la barre latérale) : aucun envoi, modèle Qwen3-VL via Ollama.
- Schéma imposé au modèle et vérifié par pydantic (`bdc/schema.py`) : une réponse hors schéma est rejetée.
- Rien n'est exporté vers le LIMS sans validation par un laborantin nommé.

### Test de non-régression

Chaque BDC validé est écrit dans `out/verite/` et enrichit le jeu de référence.

```bash
python benchmark.py --document BDC_06_ecotellus_liasse_n1
python benchmark.py --modele gemini-3.6-flash      # essai d'une autre version
```

Le score (part de champs justes) est comparé au meilleur essai précédent sur les mêmes documents : une
version n'est adoptée que si son score ne baisse pas.

| Fichier | Rôle |
|---|---|
| `config.yaml` | modèles, seuils, chemins |
| `bdc/config.py` | lecture de la configuration, du référentiel et de la clé |
| `bdc/modeles.py` | passerelle unique vers les modèles, trace des envois, bascule sur le secours |
| `bdc/schema.py` | schéma pydantic du BDC |
| `bdc/extraction_llm.py` | `lire_scan()`, `extraire_champs()`, recalage sur le référentiel |
| `bdc/controles.py` | règles de cohérence, doublons, score de confiance, alertes délai |
| `bdc/agent_relecture.py`, `bdc/agent_chat.py` | les deux agents |
| `bdc/stockage.py` | base SQLite : BDC, statut, historique des corrections |
| `bdc/export.py` | CSV LIMS et JSON des BDC validés |
| `bdc/verite.py`, `benchmark.py` | jeu de référence et test de non-régression |
| `bdc/traitement.py` | chaîne complète pour un BDC reçu |
| `bdc/gcms.py` | lien échantillon / résultats HAP du projet GC-MS (`out/gcms_resultats.csv`) |
| `app.py`, `interface/` | interface Streamlit, une page par fichier |
