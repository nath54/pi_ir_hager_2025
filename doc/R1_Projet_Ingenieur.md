
# Projet ingénieur : HAGER Group

## Implémentation et Optimisation des Réseaux de Neurones pour Microcontrôleurs

**Membres**
Cerisara Nathan : [nathan.cerisara@etu.unistra.fr](mailto:nathan.cerisara@etu.unistra.fr)
Desberg Clément : [clement.desberg@etu.unistra.fr](mailto:clement.desberg@etu.unistra.fr)
Levy Lucas : [lucas.levy2@etu.unistra.fr](mailto:lucas.levy2@etu.unistra.fr)
Jadi Ahmed Amine : [ahmed-amine.jadi@etu.unistra.fr](mailto:ahmed-amine.jadi@etu.unistra.fr)
*Lamole Etienne* ***(incertain)*** : [etienne.lamole@etu.unistra.fr](mailto:etienne.lamole@etu.unistra.fr)

**Contacts**
Hien Duc VU: [hienduc.vu@hagergroup.com](mailto:hienduc.vu@hagergroup.com)
Nicolas BRITSCH: [nicolas.britsch@hagergroup.com](mailto:nicolas.britsch@hagergroup.com)



# Table des matières {#table-des-matières}

[**Table des matières	2**](#table-des-matières)

[**1\. Introduction	3**](#1.-introduction)

[1.1. Résumé du sujet	3](#1.1.-résumé-du-sujet)

[1.2. Objectifs du projet	3](#1.2.-objectifs-du-projet)

[1.2.1. Étude des plateformes et outils de génération de code	3](#1.2.1.-étude-des-plateformes-et-outils-de-génération-de-code)

[1.2.2. Transformation des modèles Python en code embarqué	3](#1.2.2.-transformation-des-modèles-python-en-code-embarqué)

[1.2.3. Implémentation et optimisation de modèles	3](#1.2.3.-implémentation-et-optimisation-de-modèles)

[1.2.4. Simulation des modèles embarqués	3](#1.2.4.-simulation-des-modèles-embarqués)

[1.2.5. Évaluation et déploiement	3](#1.2.5.-évaluation-et-déploiement)

[1.2.6. Reporting détaillé et recommandations	3](#1.2.6.-reporting-détaillé-et-recommandations)

[**2\. Planification du travail	4**](#2.-planification-du-travail)

[2.0. Documentation	4](#2.0.-documentation)

[2.1. Conversion	4](#2.1.-conversion)

[2.1.1. Analyse et compréhension	5](#2.1.1.-analyse-et-compréhension)

[2.1.2. Conception du langage intermédiaire	6](#2.1.2.-conception-du-langage-intermédiaire)

[2.1.3. Conception d’un traducteur vers le langage intermédiaire	6](#2.1.3.-conception-d’un-traducteur-vers-le-langage-intermédiaire)

[2.1.4. Réaliser une banque de tests et vérifier que tout fonctionne bien.	6](#2.1.4.-réaliser-une-banque-de-tests-et-vérifier-que-tout-fonctionne-bien.)

[2.2. Optimisations sur ce langage intermédiaire	6](#2.2.-optimisations-sur-ce-langage-intermédiaire)

[2.2.1. Mise à l’échelle des opérations linéaires	6](#2.2.1.-mise-à-l’échelle-des-opérations-linéaires)

[2.2.2. Pruning / Suppression de poids	7](#2.2.2.-pruning)

[2.2.3. Pruning / Suppression de couche intermédiaire directement	7](#heading=h.ye887hki7gdm)

[2.2.4. Réduction de la taille d’encodage des poids (quantization)	7](#2.2.4.-réduction-de-la-taille-d’encodage-des-poids-\(quantization\))

[2.3. Optimisations sur ce langage intermédiaire	7](#2.3.-optimisations-sur-ce-langage-intermédiaire)

[**3\. Gitlab	7**](#3.-gitlab)

# 1\. Introduction {#1.-introduction}

## 1.1. Résumé du sujet {#1.1.-résumé-du-sujet}

Ce projet explore et implémente différents types et architectures de réseaux de neurones sur des microcontrôleurs, avec une attention particulière à l'optimisation et à la génération de code embarqué. L’objectif est de transformer des modèles Python développés sur des frameworks comme TensorFlow ou PyTorch en un code efficace pour des microcontrôleurs, tout en exploitant les accélérateurs matériels fournis par les fabricants. Le projet abordera également la simulation pour tester les modèles avant leur déploiement matériel, ainsi que l’évaluation des performances, des coûts et de la consommation énergétique des solutions déployées.

## 1.2. Objectifs du projet {#1.2.-objectifs-du-projet}

### 1.2.1. Étude des plateformes et outils de génération de code {#1.2.1.-étude-des-plateformes-et-outils-de-génération-de-code}

* Exploration des outils de conversion (comme TensorFlow Lite, PyTorch Mobile...).
* Compréhension des bibliothèques de déploiement spécifiques aux fabricants.

### 1.2.2. Transformation des modèles Python en code embarqué {#1.2.2.-transformation-des-modèles-python-en-code-embarqué}

* Conversion des modèles d’apprentissage automatique en code optimisé pour microcontrôleurs.
* Gestion des contraintes spécifiques (quantification, mémoire limitée).

### 1.2.3. Implémentation et optimisation de modèles {#1.2.3.-implémentation-et-optimisation-de-modèles}

* Implémentation de modèles comme CNN, RNN, MLP et Transformers.
* Compression et optimisation pour les contraintes de performance et d’énergie.

### 1.2.4. Simulation des modèles embarqués {#1.2.4.-simulation-des-modèles-embarqués}

* Utilisation de simulateurs matériels pour prédire les performances avant déploiement.
* Validation fonctionnelle des modèles sur des bancs de test virtuels.

### 1.2.5. Évaluation et déploiement {#1.2.5.-évaluation-et-déploiement}

* Analyse des performances (précision, latence, consommation d’énergie).
* Comparaison des résultats sur différentes architectures matérielles.

### 1.2.6. Reporting détaillé et recommandations {#1.2.6.-reporting-détaillé-et-recommandations}

* Documentation des étapes, des outils utilisés, des résultats obtenus et des limitations.
* Recommandations pour des améliorations futures.

# 2\. Planification du travail {#2.-planification-du-travail}

**Note :** Les tâches ne sont pas données dans un ordre chronologique, cet ordre restant encore à déterminer. Certaines parties de l’implémentation restent encore incertaines et dépendront des données du sujet.

## 2.0. Documentation {#2.0.-documentation}

Afin de permettre une maintenance, ainsi qu’une meilleure interprétabilité de ce projet, il est nécessaire de mettre une documentation à jour sur les avancées et modifications du code.
Il faudra donc documenter les idées / algorithmes, les codes et tests.

## 2.1. Conversion {#2.1.-conversion}

Un des attendus du projet étant de faire la traduction vers un langage plus optimisé. Il nous semble plus simple et plus pratique d’envisager un langage intermédiaire pour faire le lien entre le modèle dans son format d’entrée et le code final (C / assembleur). Dans cette optique, on produit un traducteur vers un format JSON (simple à charger et facilement compatible, en plus d’être à peu près compréhensible par un humain):
Fichier JSON du modèle:
{
	"type": "Model",
	"name": "…",
	"input\_shape": (… , ),
	"input\_datatype": "…",
	"output\_shape": (… , ),
	"output\_datatype": "…",
	"blocks": \[
		…	// liste de blocs, de type Block
	\]
}

Un modèle est composé de différents Blocks, avec la syntaxe suivante:
{	"type": "Block",
	"name": "…",
	"input\_shape": (… , ),
	"input\_datatype": "…",
	"output\_shape": (… , ),
	"output\_datatype": "…",
	"layers": \[
…	// liste de définitions de Tensor, d’opérations directes sur des Tensor, ou de Layers
\] }

**Note:** L’entrée d’un Block définit automatiquement un Tensor nommé X, de la forme et du type précisé, et la sortie d’un Block doit être dans un Tensor nommé Y, de la taille et du type précisés.

Représentation d’un Tensor:
{	"type": "Tensor",
	"name": "…",
	"shape": (… , ),
	"datatype": "…",
	"data": "path\_toward\_binary\_file” }

Par soucis de lisibilité et d’organisation, nous aurons chaque poids du modèle dans un fichier binaire séparé.
Ces fichiers devront être regroupés lors du script final C ou Assembleur

Représentation d’un Layer:
{	"type": "Layer",
	"name": "…",
	"tensor\_input": "…",
	"tensor\_output": "…"	}

Représentations d’opérations directes sur les tenseurs:

* Copie/Lien d’un tenseur vers un autre variable:
  {	"type": "=",
  	"tensor\_input": "…",
  	"tensor\_output": "…" }

* Opérations arithmétiques de tenseurs:
  {	"type": "+, \-, .\*, \*, ./, %, ^x",
  	"tensor\_input": "…",	// un nom de tenseur, ou une liste de tenseurs, dépendant de l’opération
  	"tensor\_output": "…" }

Ce langage intermédiaire permet de représenter les différents blocs de modèles et le flux des données, afin de pouvoir traiter, optimiser et exporter efficacement le modèle.

### 2.1.1. Analyse et compréhension {#2.1.1.-analyse-et-compréhension}

Il a fallu choisir des librairies python que le projet supportera, et analyser / comprendre les formats d’exportations des ces modèles (.h5, .keras, .torch) pour pouvoir les transformer dans ce langage intermédiaire unifié.
Pour l’instant, nous avons choisi d’abord de nous concentrer sur ***Pytorch***, et nous choisirons plus tard, si le projet supporte ***TensorFlow***.

Il nous semble donc nécessaire de lister et de documenter tous les formats d’encodage des données utilisés (ex: int8, float64, uint16, int1, …) ainsi que les fonctions / couches utilisées (ex: couche dense (tf) \= couche linéaire (pytorch), couche de convolution, …).

[Liste des couches pour langage intermédiaire](https://docs.google.com/document/d/1YU9-MRqxcMLJRASWjCxdkN-StsMlnh8Jp-h2MPxlIJo/edit?usp=sharing)

### 2.1.2. Conception du langage intermédiaire {#2.1.2.-conception-du-langage-intermédiaire}

Pour pouvoir traduire le modèle de base vers un langage intermédiaire adapté, il est essentiel de concevoir ce langage après avoir analysé et listé les différentes possibilités.

Le premier langage intermédiaire choisi et qu’on utilisera en premier lieu pour la suite du projet est le design JSON qui à été présenté ci-dessus.

### 2.1.3. Conception d’un traducteur vers le langage intermédiaire {#2.1.3.-conception-d’un-traducteur-vers-le-langage-intermédiaire}

 Un des premiers problèmes que nous avons rencontré est le manque d'informations. Le(s) fichier(s) de poids de modèles exportés par Pytorch / Tensorflow ne contiennent pas forcément la structure du modèle, mais seulement les matrices de poids. Il va falloir soit prendre en entrée directement la structure du modèle, ce qui est trop demandant, et pas flexible du tout, ou bien faire un script qui va directement charger le code python qui contient la définition du modèle, et le compiler pour en obtenir le code intermédiaire.

### 2.1.4. Réaliser une banque de tests et vérifier que tout fonctionne bien. {#2.1.4.-réaliser-une-banque-de-tests-et-vérifier-que-tout-fonctionne-bien.}

Dans l'objectif d’avoir un code le plus juste (en termes d’accuracy par rapport au modèle de base) et optimisé possible. Il faut créer et documenter une banque de tests qui permet à la fois de vérifier la justesse et la vitesse du code mais également de prouver son bon fonctionnement.
Pour cela, il faut que ces tests soient les plus variés et donc prouvent le bon fonctionnement du plus de fonctionnalités possible.

## 2.2. Optimisations sur ce langage intermédiaire {#2.2.-optimisations-sur-ce-langage-intermédiaire}

### 2.2.1. Mise à l’échelle des opérations linéaires {#2.2.1.-mise-à-l’échelle-des-opérations-linéaires}

Commençons par un exemple:

linearLayer(512, 1024\)
	linearLayer(1024, 256\)

Peut être optimisé en:

	linearLayer(512, d)
	linearLayer(d, 256\)

Avec d inférieur à 1024.
Nous aurons besoin d’une étude de l’évolution des pertes de performances en fonction de d.
Nous aurons également besoin de savoir comment calculer les matrices de poids des couches optimisées en fonction des anciennes matrices de poids.

### 2.2.2. Pruning {#2.2.2.-pruning}

### 2.2.4. Réduction de la taille d’encodage des poids (quantization) {#2.2.4.-réduction-de-la-taille-d’encodage-des-poids-(quantization)}

Nous devrons étudier les pertes de performances en fonction de la quantization appliquée au modèle (ex: float64 \-\> int16, ou float64 \-\> int4)

### 2.2.5. Distillation

## 2.3. Optimisations sur ce langage intermédiaire {#2.3.-optimisations-sur-ce-langage-intermédiaire}

Une fois le traducteur en langage intermédiaire fait, nous pourrons chercher des optimisations supplémentaires dans différents objectifs : utiliser moins de RAM, supprimer les calculs inutiles, détecter les calculs redondants, essayer de réduire le plus possible le nombre d’opérations arithmétiques, …

# 3\. Gitlab {#3.-gitlab}

Nous avons créé un projet gitlab: [https://gitlab.unistra.fr/cerisara/pi\_hager\_2025](https://gitlab.unistra.fr/cerisara/pi_hager_2025) sur lequel nous avons commencé à organiser la structure du travail.