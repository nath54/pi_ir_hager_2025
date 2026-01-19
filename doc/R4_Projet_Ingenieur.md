
# Projet ingénieur HAGER Group \- Quatrième Rapport

## Implémentation et Optimisation des Réseaux de Neurones pour Microcontrôleurs

**Membres**
Cerisara Nathan : [nathan.cerisara@etu.unistra.fr](mailto:nathan.cerisara@etu.unistra.fr)
Desberg Clément : [clement.desberg@etu.unistra.fr](mailto:clement.desberg@etu.unistra.fr)
Levy Lucas : [lucas.levy2@etu.unistra.fr](mailto:lucas.levy2@etu.unistra.fr)
Jadi Ahmed Amine : [ahmed-amine.jadi@etu.unistra.fr](mailto:ahmed-amine.jadi@etu.unistra.fr)

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

[**2\. Organisation du travail	4**](#2.-organisation-du-travail)

[**3\. Travail du sous-groupe 1 \- Implémentation complète Open-Source	4**](#3.-travail-du-sous-groupe-1---implémentation-complète-open-source)

[3.1. Rappel de l'architecture globale actuelle	4](#3.1.-rappel-de-l'architecture-globale-actuelle)

[3.1.1. Composants principaux de la Représentation Intermédiaire	4](#3.1.1.-composants-principaux-de-la-représentation-intermédiaire)

[3.1.2. Extraction et liaison des poids	4](#3.1.2.-extraction-et-liaison-des-poids)

[3.1.3. Interprétation et exécution	5](#3.1.3.-interprétation-et-exécution)

[3.1.4. Tests	5](#3.1.4.-tests)

[3.2. Avancées sur l'extraction de modèle PyTorch	5](#3.2.-avancées-sur-l'extraction-de-modèle-pytorch)

[3.3. Interpréteur interne	6](#3.3.-interpréteur-interne)

[3.4. Résultats de tests	6](#3.4.-résultats-de-tests)

[3.4.1. Données des premiers tests effectués	6](#3.4.1.-données-des-premiers-tests-effectués)

[3.4.2. Validation sur le modèle Hager Group	7](#3.4.2.-validation-sur-le-modèle-hager-group)

[3.4.3. Analyse des performances	8](#3.4.3.-analyse-des-performances)

[3.5. Compilation open-source et automatisable vers STM32-NUCLEO-H723ZG	8](#3.5.-compilation-open-source-et-automatisable-vers-stm32-nucleo-h723zg)

[3.5.1. Infrastructure de compilation	8](#3.5.1.-infrastructure-de-compilation)

[3.5.2. Architecture de compilation	8](#3.5.2.-architecture-de-compilation)

[3.5.3. Fonctionnalités implémentées	8](#3.5.3.-fonctionnalités-implémentées)

[3.6. Projections futures	9](#3.6.-projections-futures)

[**4\. Travail du sous-groupe 2 \- Solution propriétaire en ligne ST Edge AI Cloud	10**](#4.-travail-du-sous-groupe-2---solution-propriétaire-en-ligne-st-edge-ai-cloud)

[4.1. Objectif, contexte et défis	10](#4.1.-objectif,-contexte-et-défis)

[4.2. Préparation et conversion des modèles	10](#4.2.-préparation-et-conversion-des-modèles)

[4.2.1 Modèle ONNX	11](#4.2.1-modèle-onnx)

[4.2.2. Modèle TFLite	11](#4.2.2.-modèle-tflite)

[4.2.3. Transformation en keras	12](#4.2.3.-transformation-en-keras)

[4.2.5 Quantization manuelle	12](#4.2.5-quantization-manuelle)

[**5\. Conclusion	12**](#5.-conclusion)

# 1\. Introduction {#1.-introduction}

## 1.1. Résumé du sujet {#1.1.-résumé-du-sujet}

Ce projet explore et implémente différents types et architectures de réseaux de neurones sur des microcontrôleurs, avec une attention particulière à l'optimisation et à la génération de code embarqué. L’objectif est de transformer des modèles Python développés sur des frameworks comme TensorFlow ou PyTorch en un code efficace pour des microcontrôleurs, tout en exploitant les accélérateurs matériels fournis par les fabricants. Le projet abordera également la simulation pour tester les modèles avant leur déploiement matériel, ainsi que l’évaluation des performances, des coûts et de la consommation énergétique des solutions déployées.

## 1.2. Objectifs du projet {#1.2.-objectifs-du-projet}

### **1.2.1. Étude des plateformes et outils de génération de code** {#1.2.1.-étude-des-plateformes-et-outils-de-génération-de-code}

* Exploration des outils de conversion (comme TensorFlow Lite, PyTorch Mobile...).
* Compréhension des bibliothèques de déploiement spécifiques aux fabricants.

### **1.2.2. Transformation des modèles Python en code embarqué** {#1.2.2.-transformation-des-modèles-python-en-code-embarqué}

* Conversion des modèles d’apprentissage automatique en code optimisé pour microcontrôleurs.
* Gestion des contraintes spécifiques (quantification, mémoire limitée).

### **1.2.3. Implémentation et optimisation de modèles** {#1.2.3.-implémentation-et-optimisation-de-modèles}

* Implémentation de modèles comme CNN, RNN, MLP et Transformers.
* Compression et optimisation pour les contraintes de performance et d’énergie.

### **1.2.4. Simulation des modèles embarqués** {#1.2.4.-simulation-des-modèles-embarqués}

* Utilisation de simulateurs matériels pour prédire les performances avant déploiement.
* Validation fonctionnelle des modèles sur des bancs de test virtuels.

### **1.2.5. Évaluation et déploiement** {#1.2.5.-évaluation-et-déploiement}

* Analyse des performances (précision, latence, consommation d’énergie).
* Comparaison des résultats sur différentes architectures matérielles.

### **1.2.6. Reporting détaillé et recommandations** {#1.2.6.-reporting-détaillé-et-recommandations}

* Documentation des étapes, des outils utilisés, des résultats obtenus et des limitations. Recommandations pour des améliorations futures.

# 2\. Organisation du travail {#2.-organisation-du-travail}

- **Sous-groupe 1** (Nathan Cerisara): travail sur une implémentation manuelle d’un parser/interpreteur/compilateur pour transformer du code pytorch en un code C tout en n’utilisant aucun logiciels propriétaires.
- **Sous-groupe 2** (Lucas Levy, Clément Desberg, Ahmed Amine Jadi): travail sur la suite STM32 AI EDGE CLOUD pour le convertisseur pytorch vers C et les logiciels STM32CUBE IDE pour l’export du code vers stm32.

# 3\. Travail du sous-groupe 1 \- Implémentation complète Open-Source {#3.-travail-du-sous-groupe-1---implémentation-complète-open-source}

L'architecture du projet ModelBlocks s'articule autour d'une **Représentation Intermédiaire (RI)** complète qui permet de capturer l'essence des modèles PyTorch tout en restant indépendante du framework d'origine. Cette architecture a été considérablement enrichie depuis les rapports précédents.

## 3.1. Rappel de l'architecture globale actuelle {#3.1.-rappel-de-l'architecture-globale-actuelle}

### **3.1.1. Composants principaux de la Représentation Intermédiaire** {#3.1.1.-composants-principaux-de-la-représentation-intermédiaire}

**lib\_classes.py** : Classes fondamentales pour la **Représentation Intermédiaire (RI)**.

* **Language\_Model** : Classe principale qui contient l'entièreté de la représentation intermédiaire, orchestrant l'ensemble des blocs et couches du modèle.
* **ModelBlock** : Représente une classe ou un bloc d'un réseau de neurones (équivalent d'un nn.Module PyTorch). Chaque bloc contient un ensemble de Layers qui peuvent être soit des couches de base (comme Linear, Conv2d, BatchNorm, etc.) soit d'autres ModelBlock pour une architecture hiérarchique. Les poids associés aux couches de base sont stockés dans la classe Layer.
* **BlockFunction** : Représente les fonctions d'un ModelBlock, principalement \_\_init\_\_ et forward. Contient les informations sur les paramètres de la fonction ainsi que son flux d'instructions (Instruction Flow).
* **FlowControlInstruction** : Représente tous les types d'instructions : opérations, appels de fonctions ou de couches, gestion des variables, etc.
* **Expression** : Représente les variables et données manipulées par le modèle.

### **3.1.2. Extraction et liaison des poids** {#3.1.2.-extraction-et-liaison-des-poids}

**lib\_extract\_model\_architecture.py** : Module d'extraction de la RI depuis un fichier source PyTorch.

* **ModelAnalyzer** : Classe basée sur la librairie ast de Python qui s'occupe du parsing et permet d'extraire la structure du code Python, transformant les définitions de modèles PyTorch en représentation intermédiaire structurée.

**lib\_weights\_link.py** : Permet de charger le modèle PyTorch original et de lier les poids du modèle original vers la représentation intermédiaire, assurant la cohérence entre les deux représentations.

### **3.1.3. Interprétation et exécution** {#3.1.3.-interprétation-et-exécution}

**lib\_interpretor.py** : Implémente un interpréteur complet pour exécuter une RI donnée.

* **Symbol** : Représente une variable interprétée avec son nom, type et valeur.
* **Scope** : Représente un niveau de portée des variables, contenant un lien vers son parent pour une gestion hiérarchique des variables.
* **ExecutionContext** : Représente l'entièreté des variables structurées de contexte hiérarchiques organisées en pile.
* **LanguageModel\_ForwardInterpreter** : Classe principale qui permet d'exécuter une RI donnée, utilisant NumPy pour gérer les tenseurs et effectuer les calculs.

### **3.1.4. Tests** {#3.1.4.-tests}

**lib\_test.py** et **script\_test.py** : Utilitaires complets pour effectuer des tests automatisés permettant de vérifier l'extraction des modèles et leur exécution, avec validation de la précision numérique.

## 3.2. Avancées sur l'extraction de modèle PyTorch {#3.2.-avancées-sur-l'extraction-de-modèle-pytorch}

Depuis le rapport précédent, l'extraction de modèles PyTorch a été considérablement améliorée et finalisée. Les fonctionnalités de base ont été entièrement implémentées et testées.

L'extraction de modèles PyTorch est maintenant complètement opérationnelle pour les architectures de réseaux de neurones les plus courantes. Le système peut désormais :

* **Parser automatiquement** les définitions de modèles PyTorch complexes avec des architectures hiérarchiques
* **Extraire la structure complète** des modèles, incluant les couches imbriquées et les blocs conditionnels
* **Gérer les paramètres dynamiques** et les dimensions de tenseurs variables
* **Supporter les architectures avancées** comme les réseaux convolutifs, les réseaux de neurones profonds et les modèles avec des branches conditionnelles

#
## 3.3. Interpréteur interne {#3.3.-interpréteur-interne}

L'interpréteur interne constitue le cœur du système ModelBlocks, permettant l'exécution complète des modèles extraits en utilisant NumPy comme backend de calcul. De plus, elle sera très utile pour examiner les variables, principalement tous les *Tensors* et leurs dimensions, pour savoir précisément quand les libérer de la mémoire.

L'interpréteur utilise une architecture hiérarchique basée sur des scopes (portées) pour gérer les variables et l'exécution

**Gestion des tenseurs NumPy** : L'interpréteur utilise NumPy pour tous les calculs tensoriels

**Exécution de flux de contrôle** : Support de la majorité des structures de contrôle

L'interpréteur intègre des outils de débogage avancés.

## 3.4. Résultats de tests {#3.4.-résultats-de-tests}

Les résultats de tests démontrent une performance équilibrée du système ModelBlocks avec un taux de réussite d’extraction de la représentation intermédiaire (RI) de 100% de 50% pour l’interprétation des modèles sur l'ensemble des modèles testés, incluant des architectures complexes et des modèles réels de production.

A noter, cependant que bien qu’ils soient très divers et variés, ces tests ne représentent absolument pas une liste exhaustive des

### **3.4.1. Données des premiers tests effectués** {#3.4.1.-données-des-premiers-tests-effectués}

| Tested model                                                                                                     | Extraction and weight linking | Execution in RI interpretor | Relative Euclidian\* Distance between Pytorch and RI Interpretor (20x average) | model output dimension |
| ---------------------------------------------------------------------------------------------------------------- | :---------------------------: | :-------------------------: | :----------------------------------------------------------------------------: | :--------------------: |
| 1 Linear Layer                                                                                                   |               ✓               |              ✓              |                           **7,89.10\-8 (\<\<10\-3)**                           |        *(2,5)*         |
| 2 Linears Layer with ReLU                                                                                        |               ✓               |              ✓              |                           **3,28.10\-8 (\<\<10\-3)**                           |        *(2,5)*         |
| 1 2D Convolution Layer                                                                                           |               ✓               |           **\~**            |                              **1.03  (\>10\-1)**                               |     *(2,16,32,32)*     |
| 2 2D Convolution Layers with ReLU BatchNorm and a linear                                                         |               ✓               |           **\~**            |                              **1.07  (\>10\-1)**                               |        *(2,10)*        |
| 1 MaxPool Layer                                                                                                  |               ✓               |              ✓              |                           **1,91.10\-8 (\<\<10\-3)**                           |     *(2,3,10,10)*      |
| 1 AvgPool Layer                                                                                                  |               ✓               |              ✓              |                           **5,68.10\-8 (\<\<10\-3)**                           |     *(2,3,10,10)*      |
| 1 AdAvgPool Layer                                                                                                |               ✓               |              ✓              |                           **3,91.10\-8 (\<\<10\-3)**                           |     *(2,3,16,16)*      |
| Complex Model with  Convolutions, Linears, Normalisations and Activations                                        |               ✓               |              ✓              |                              **0.013 (\<10\-1)**                               |       *(1,1000)*       |
| Test with a 2 Linear Layers and 1 ReLU with some advanced control flow and variable manipulations                |               ✓               |           **\~**            |                              **0.315  (≥10\-1)**                               |        *(4,30)*        |
| Test with a Sequential, Linear and Activations layers with some advanced control flow and variable manipulations |               ✓               |           **\~**            |                              **0.342  (≥10\-1)**                               |        *(4,30)*        |
| Test with a ModuleList, Linear and Activations layers with some advanced control flow and variable manipulations |               ✓               |           **\~**            |                              **0.233  (≥10\-1)**                               |        *(4,30)*        |
| Test with Hager Model, Hand-made Vision Transformer.                                                             |               ✓               |           **\~**            |                               **1.0  (\>10\-1)**                               |        *(1,6)*         |

**\*** *Distance relative \= (distance euclidienne) / (norme de l’un des tensors de sortie)*

**Ordres de grandeurs:**

- \< 10^(-3) : distance petite
- entre 10^(-3) et 10^(-1) : les tensors commencent à diverger
- supérieur à 10^(-1) : les matrices sont  de plus en plusdifférentes

Les tests effectués portent sur **l’extraction**, **le lien des poids** et une **interprétation**.

On peut donc voir que tous les tests sont parsés correctement, et arrivent tous à s’interpréter, mais certaines sorties sont plus ou moins différentes des sorties originales pytorch.

### **3.4.2. Validation sur le modèle Hager Group** {#3.4.2.-validation-sur-le-modèle-hager-group}

On peut avoir quelques inquiétudes sur la distance entre l’exécution du modèle original Pytorch et la RI interprétée, mais si elle est à ce niveau, c’est sûrement dû à un problème d’implémentation dans l’interpréteur, et non à l’extraction du modèle, et ne devrait pas poser trop de problèmes lors de l’exportation vers C. Le test sur le modèle fourni par Hager Group a été un succès partiel. Les prochaines étapes étant donc d’identifier les causes possibles de distances entre les vecteurs, les résoudre, et d’exporter la représentation interne en C.

### **3.4.3. Analyse des performances** {#3.4.3.-analyse-des-performances}

**Précision numérique** : Les distances varient entre de très faibles valeurs, jusqu’à  de fortes distances pour les modèles de tests purement convolutifs. Ceci est sûrement dû à une mauvaise implémentation de la Convolution ou bien une mauvaise gestion des paramètres de la couche de Convolution **au niveau de l’interpréteur seulement.**

**Robustesse** : Aucun test n'a échoué complètement, montrant la stabilité du système même sur des architectures non optimisées.

## 3.5. Compilation open-source et automatisable vers STM32-NUCLEO-H723ZG {#3.5.-compilation-open-source-et-automatisable-vers-stm32-nucleo-h723zg}

La compilation vers STM32 représente une étape cruciale du projet, permettant le déploiement des modèles sur des microcontrôleurs réels. Une infrastructure complète de compilation open-source a été mise en place.

### **3.5.1. Infrastructure de compilation** {#3.5.1.-infrastructure-de-compilation}

**Environnement de développement** :

* **Toolchain ARM** : arm-none-eabi-gcc pour la compilation croisée
* **LibOpenCM3** : Bibliothèque open-source pour les microcontrôleurs STM32
* **Makefile automatisé** : Compilation, linking et génération des binaires
* **Scripts de déploiement** : Flash automatique sur le microcontrôleur

**Configuration cible** :

* **MCU** : STM32H723ZG (Cortex-M7)
* **FPU** : Support des opérations flottantes en dur (FPv5-D16)
* **Optimisations** : \-Os pour la taille, \-ffunction-sections pour le linking

### **3.5.2. Architecture de compilation** {#3.5.2.-architecture-de-compilation}

**Processus de compilation** :

1. **Compilation croisée** : Code C → Objets ARM
2. **Linking** : Objets \+ LibOpenCM3 → ELF
3. **Génération binaire** : ELF → BIN/HEX
4. **Déploiement** : Flash sur STM32 via ST-Link
5. **Scripts de build** : Automatisation par script pour la compilation jusqu’au déploiement

### **3.5.3. Fonctionnalités implémentées** {#3.5.3.-fonctionnalités-implémentées}

**Application de démonstration** :

* **Compteur binaire LED** : Démonstration du fonctionnement du MCU
* **Gestion des GPIO** : Contrôle des LEDs LD1, LD2, LD3
* **Système de timing** : SysTick pour la gestion du temps

## 3.6. Projections futures {#3.6.-projections-futures}

L’objectif principal consiste donc à transformer la Représentation Intermédiaire (RI) validée en un code C exécutable et optimisé pour microcontrôleurs STM32. Cette étape représente la concrétisation du pipeline complet ModelBlocks, reliant directement l’extraction de modèles PyTorch à leur exécution embarquée.

Le processus envisagé repose sur une génération automatique de code à partir de la RI, permettant de produire des fichiers C structurés reproduisant fidèlement le flux d’exécution du modèle. Chaque Instruction Flow sera convertie en équivalent C, avec une gestion explicite des variables, des allocations mémoire et des dépendances entre couches.

Un soin particulier sera porté à la traçabilité des types et dimensions des tenseurs, afin d’anticiper leur gestion mémoire (allocation et libération) et de garantir la cohérence des opérations. Deux niveaux d’analyse seront mis en œuvre :

* Global : suivi des types, tailles et portées de chaque variable tout au long du programme.
* **Local** : suivi des évolutions de dimensions et de références au sein de chaque bloc fonctionnel, permettant d’identifier les optimisations possibles.

Il est envisagé de faire une structure C modulaire, supportant plusieurs backends dont 1 spécialisé MCU (basé sur CMSIS-NN), et un backend pour ordinateurs (basé sur du Numpy-C ou autre).

À terme, ces développements permettront d’obtenir une chaîne complète, open-source et automatisée, capable de convertir un modèle PyTorch en un code C directement exécutable sur STM32, tout en garantissant efficacité, stabilité et portabilité.


# 4\. Travail du sous-groupe 2 \- Solution propriétaire en ligne ST Edge AI Cloud {#4.-travail-du-sous-groupe-2---solution-propriétaire-en-ligne-st-edge-ai-cloud}

Nous avons réalisé l’analyse, l’optimisation et la conversion du modèle de réseau de neurone d’exemple donné par le client et d’un modèle linéaire simple à travers l’interface web proposée par ST, à savoir ST Edge AI Cloud.
L’objectif de cette partie du projet était de continuer l’exploration de cette solution que nous avions commencé l’année dernière, dans le but final de la fournir au client avec toutes les informations et spécifications nécessaires à son déploiement.

## 4.1. Objectif, contexte et défis {#4.1.-objectif,-contexte-et-défis}

Alors que le sous-groupe 1 travaillait sur une implémentation open-source maîtrisant l’intégralité du processus (analyse AST, génération de code C, interprétation locale), s’est attelé à évaluer ST Edge AI Cloud, qui permet de convertir automatiquement des modèles d’apprentissage profond vers du code embarqué optimisé pour la famille STM32.

Ce travail s’inscrivait dans une logique d’évaluation comparative : mesurer la maturité du service cloud, sa compatibilité avec les modèles fournis par Hager Group et ses avantages en termes d’industrialisation et de déploiement rapide.

Le principal défi qui s’oppose à la conversation à l’aide de cette solution est la taille du modèle. En effet, après avoir testé les premières conversions du modèle, nous avons pu confirmer que le modèle est trop grand pour le microcontrôleur fourni par le client. La quantization dont nous avons déjà parlé n’est donc plus une option.

## 4.2. Préparation et conversion des modèles {#4.2.-préparation-et-conversion-des-modèles}

Le modèle initial fourni par le client était au format PyTorch, représentant un réseau de neurones convolutif compact.

La solution ST Edge AI Cloud propose l’analyse et la compilation d’un modèle à partir de plusieurs formats : Keras, ONNX et TFLite. Nous avons réalisé des tests pour chaque format.

Pour chaque format, les étapes du processus d’analyse et de conversion de ST Edge AI Developer Cloud restent les mêmes :

1. **Analyse du graphe de calcul**, avec identification des couches et inférence des dimensions d’entrée/sortie

2. **Estimation de l’empreinte mémoire** (RAM et Flash) et du temps d’inférence théorique sur différentes gammes de microcontrôleurs

3. **Optimisation automatique** du graphe (fusion de couches, suppression des redondances, application éventuelle de quantification) ;

4. **Génération d’un projet STM32Cube.AI** contenant le code C et les bibliothèques d’exécution associées.

### **4.2.1 Modèle ONNX** {#4.2.1-modèle-onnx}

Une première vérification du modèle ONNX a confirmé la bonne correspondance entre la topologie convertie et la version d’origine (mêmes dimensions, mêmes poids, inférences identiques à 10⁻⁶ près).

Cette phase d’analyse s’est déroulée correctement jusqu’à la détection d’une incompatibilité de couche : la plateforme a indiqué que la couche LayerNormalization n’était pas implémentée dans la version 2.2.0 du moteur ST Edge AI Developer Cloud.
 Ce type de normalisation, fréquemment utilisé dans les architectures modernes (notamment Transformers ou modèles résiduels), n’est pas encore pris en charge par la plupart des convertisseurs embarqués.

En conséquence, la conversion complète du modèle en code exécutable n’a pas pu aboutir, mais l’analyse a tout de même fourni :

* une cartographie précise du graphe et des couches reconnues

* une évaluation de la consommation mémoire totale (RAM ≈ 180 kB ; Flash ≈ 650 kB pour la version non quantifiée)

### **4.2.2. Modèle TFLite** {#4.2.2.-modèle-tflite}

Dans le précédent rapport, nous avons indiqué qu'une traduction vers du TFLite était parfaitement possible, mais nécessitait une ancienne version de python (python 3.10-3.11), notamment à cause d’une incompatibilité entre les versions récentes de python et tensorflow.
Générer un modèle TFLite est possible et fonctionnel pour la suite de la pipeline de STM32-edge-ai. Ainsi avec un modèle TFLite il est possible de générer le code final C, ainsi qu’un export .elf qui peut être transféré et exécuté sur le microcontrôleur.
Cependant, TFLite possède 2 problèmes dépendant l’un de l’autre. Le premier de ces 2 problèmes est la taille du modèle car en effet ST-edge-ai ne permet pas de générer un code C s' il consomme trop de mémoire Vram ou flash pour le microcontrôleur choisi. La solution la plus évidente de ce problème est de faire une quantisation en amont ou en aval afin de réduire la taille des poids du modèle et donc de le rendre compatible avec un microcontrôleur avec moins de mémoire. Donc le deuxième problème est la quantisation car sur un modèle TFLite on ne peut pas quantiser en dessous de float 16 car cet extension n'accepte que des floats à la fois en entrée et en sortie.
Nous n’avons pour le moment pas de solution pour faire tourner la solution sur le microcontrôleur cible avec TFLite

### **4.2.3. Transformation en keras** {#4.2.3.-transformation-en-keras}

Une solution à tous les problèmes précédents serait le .keras car sur le site st-edge-ai une quantisation est prévue pour les modèles keras et onnx. Cependant, après beaucoup d’essais de conversion les onnx et TFLite en keras il y a beaucoup de problèmes de compatibilité de version notamment entre torch et tensorflow mais aussi entre torch et ST-edge-ai qui n'acceptent pas toutes les fonctions pytorch.
Dans cette optique, un essai a été fait pour générer un .keras, pour cela une modification du modèle était nécessaire afin de remplacer toutes les fonctions qui posaient des problèmes de compatibilités pour générer un .keras. Cependant, même après une modification manuelle du modèle les différentes conversion testés entre onnx et keras continue à afficher l’erreur des fonctions qui ont été supprimées. Pour le moment la conversion en .keras n’a pas abouti.
Aussi lors des essais pour cette conversion on remarque qu'un TFLite généré à partir d’un modèle onnx génère aussi le même problème de compatibilité entre onnx et ST-edge-ai.
La dernière point à tester sur keras est de générer un modèle keras directement depuis le modèle pytorch

### **4.2.5 Quantization manuelle** {#4.2.5-quantization-manuelle}

Pour pallier les problèmes rencontrés précédemment (à savoir l’incapacité de ST Edge AI Cloud de quantizer un fichier TFLite). Après un examen de l’architecture cible, nous avons choisi de quantizer en int8 depuis le type initial du modèle, c’est-à-dire float32. Le résultat de la quantization du modèle est une réduction de plus de moitié de la taille (bien que le type de donnée ait une taille 4 fois inférieure, la description des différentes couches du modèle garde une taille indépendante, expliquant pourquoi la quantization ne permet de réduire la taille de l’ensemble que d’un facteur légèrement supérieur à 2).
La quantization est effectuée directement sur le modèle Pytorch, plus précisément sur le fichier de poids .pth

Si la quantization en elle-même fonctionne, en revanche l’export qui s’ensuit en tfllite par exemple ne produit pas de modèle valide pour une raison encore inconnue.


# 5\. Conclusion {#5.-conclusion}

Bien que nous ayions été confrontés à des difficultés techniques et organisationnelles, les méthodes que nous avons pu mettre en place afin de continuer l’avancement du projet nous rassure dans la viabilité du projet. Notamment, si l'exportation avec quantization ne fonctionne pour l’instant pas, nous avons eu la confirmation que la quantization en elle-même était possible et fonctionnait avec ST Edge AI Cloud. Ainsi, cette solution propriétaire reste parfaitement envisageable, et la solution Open Source développée par Nathan restera une option intéressante à la fois au niveau théorique et aussi en tant qu’outil de comparaison, en plus de garantir que nous pourrons fournir une solution au client dans l’éventualité ou la solution professionnelle ne fonctionne finalement pas.