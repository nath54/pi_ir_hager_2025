
# Projet ingénieur HAGER Group \- Troisième Rapport

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

[**2\. Etat de l’art	4**](#2.-etat-de-l’art)

[2.1. Executorch	4](#2.1.-executorch)

[2.2 LiteRT (anciennement Tensorflow Lite)	4](#2.2-litert-\(anciennement-tensorflow-lite\))

[2.3 Torchscript	5](#2.3-torchscript)

[2.4 ST Edge AI Developer Cloud	6](#2.4-st-edge-ai-developer-cloud)

[**3\. Changement de méthode principale et de répartition du travail	7**](#3.-changement-de-méthode-principale-et-de-répartition-du-travail)

[**4\. Conclusion	7**](#4.-conclusion)

[**5\. Références	8**](#5.-références)

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

# 2\. Etat de l’art {#2.-etat-de-l’art}

Afin d’avoir une vision globale de ce qu’il était possible de faire avec des solutions préexistantes, nous avons dressé une liste de logiciel et programme de traduction de pytorch en c++. Pour chacune d’entre elles, nous avons examiné les paramètres de cette solution afin de déterminer si elle correspondait ou non à nos objectifs.

## 2.1. Executorch {#2.1.-executorch}

Executorch est un framework d'exécution pour PyTorch spécifiquement conçu pour le déploiement de modèles sur des environnements contraints tels que les appareils mobiles et l'embarqué, y compris les microcontrôleurs. Lancé par l'équipe PyTorch, il vise à fournir une solution portable, efficace et optimisée en mémoire pour l'exécution d'inférences sur des dispositifs à faibles ressources. Il permet de convertir des modèles PyTorch en un format intermédiaire (\`.pte\`) optimisé pour l'exécution sur diverses architectures matérielles. Plus d'informations sont disponibles sur le site officiel de PyTorch Executorch [\[1\]](#bookmark=id.b6zsh3z4m5kx) et la documentation [\[2\]](#bookmark=id.d39i1xkanyyb).

**Difficultés rencontrées:**

Lors de l'évaluation de Executorch, plusieurs problèmes ont été rencontrés qui ont entravé son utilisation dans le cadre de ce projet.

Premièrement, une dépendance logicielle spécifique a nécessité l'utilisation d'une version de Python potentiellement obsolète (3.10) pour l'une des bibliothèques requises, compliquant l'environnement de développement.

De plus, les tentatives d'exécution ou de conversion via les scripts fournis n'ont pas abouti à cause de multiples complications, empêchant une évaluation fonctionnelle de cette approche.

**Raisons du rejet :**

La difficulté de la mise en place d’une installation, ainsi que les nombreuses dépendances nécessaires au développement nous ont poussés à écarter cette option.

## 2.2 LiteRT (anciennement Tensorflow Lite) {#2.2-litert-(anciennement-tensorflow-lite)}

Lite RunTime (anciennement Tensorflow Lite) est un ensemble d'outils développé par Google pour permettre le déploiement de modèles d'apprentissage automatique sur des appareils mobiles, embarqués et IoT. La branche "LiteRT for Microcontrollers" (LiteRT Micro) est spécifiquement optimisée pour les microcontrôleurs avec des contraintes extrêmes en termes de mémoire (quelques kilo-octets) et de puissance de calcul. Le processus implique généralement la conversion du modèle original (souvent TensorFlow/Keras, mais d'autres formats comme ONNX peuvent être supportés via des convertisseurs) en un format \`.tflite\` optimisé, suivi de l'intégration avec une bibliothèque d'exécution minimaliste adaptée à la cible matérielle. Nous avons accès à la documentation officielle [\[3\]](#bookmark=id.w59uxolpir68) et à la documentation spécifique pour les microcontrôleurs [\[4\]](#bookmark=id.gso9j55a0uy3).

**Difficultés rencontrées:**

L'évaluation de LiteRT pour la conversion de modèles a été partiellement réussie. La production du modèle au format \`.tflite\` a pu être menée à bien. Cependant, des problèmes de compatibilité logicielle similaires à ceux rencontrés avec d'autres outils (nécessitant potentiellement Python 3.10) ont été observés pour certaines étapes. Le blocage principal est survenu lors de la compilation du projet C++ visant à intégrer le modèle \`.tflite\`, qui s'est soldée par des erreurs de compilation non résolues, empêchant le test du modèle sur la cible matérielle ou en simulation C++.

**Résultats:**

A cause d’erreurs de compilation du modèle converti en C++ par LiteRT, cette conversion n’est pas utilisable. Cependant, la conversion en .tflite fonctionne et est utilisable pour une potentielle pré-conversion.

## 2.3 Torchscript {#2.3-torchscript}

TorchScript est une fonctionnalité de PyTorch qui permet de créer des modèles sérialisables et optimisables à partir de code PyTorch. Il s'agit d'un sous-ensemble du langage Python qui peut être compilé et exécuté indépendamment de l'interpréteur Python, ce qui le rend adapté au déploiement dans des environnements de production, notamment en C++. TorchScript convertit les modèles en une représentation graphique qui peut ensuite être exécutée par la bibliothèque C++ de PyTorch. La documentation officielle [\[5\]](#bookmark=id.lhgu1ku3ciaf) de TorchScript est bien fournie.

**Difficultés rencontrées:**

Les tentatives d'utilisation de TorchScript pour le déploiement C++ ont rencontré des difficultés significatives. Bien que l'exportation du modèle au format \`.pt\` (TorchScript) ait été fonctionnelle, la compilation du projet C++ intégrant le modèle n'a pas abouti, produisant des erreurs de compilation similaires à celles rencontrées avec TensorFlow Lite. De plus, des problèmes liés aux versions des dépendances Python (nécessitant potentiellement Python 3.10) ont pu contribuer à ces difficultés lors de l'étape de conversion ou de préparation.

**Raison du rejet:**

Nous choisissons d’écarter cette option pour des raisons de difficulté de maintenance et de compatibilité. Le projet n’étant plus maintenu et reposant sur des librairies dépréciées, il nous semble peu pertinent de choisir cette option.

## 2.4 ST Edge AI Developer Cloud {#2.4-st-edge-ai-developer-cloud}

Dans la mesure où la solution est propriétaire, nous avons commencé par vérifier avec le client la possibilité de l’utiliser.
Cette solution consiste en une solution propriétaire édité par ST, d’une part sous la forme d’un logiciel cloud [\[6\]](#bookmark=id.mqsczead2mbd) pour l’optimisation et la génération du code c, puis accompagné d’un IDE [\[7\]](#bookmark=id.t3d74wjl6taq) pour l’édition du code final et l’exportation sur le microcontrôleur STM32.

Le site web ST Edge AI Developer Cloud est théoriquement censé supporter les modèles exportés en ONNX depuis Pytorch, mais nous avions eu une erreur liée au non support des couches “LayerNorm” provenant de Pytorch. En revanche, si nous convertissions le modèle Pytorch en format TFLite, il était bien compatible avec le site web ST Edge AI Developer Cloud.

Nous avons pu ensuite tester la pipeline complète (sauf quantization car apparemment non supportée sur ce modèle sans explications supplémentaires, merci les logiciels propriétaires), et les résultats qu’ils donnent indiquent \~75 ms de temps d’inférence pour une entrée.
Ces tests indiquent aussi qu’il faudra rajouter une mémoire flash sur le microprocesseur.
De plus, dans la pipeline, les modèles possibles de MCU STM32 donnés n’étaient pas disponibles, mais nous avons fait les tests sur un modèle qui avait l’air très proche (stm32h725ag \-\>  STM32H735G-DK).

Nous avons aussi comparé sur ST Edge AI Developer Cloud avec des processeurs stm32 qui contiennent un processeur de type Cortex-A et qui ont plus de RAM, et nous avons eu des temps d'exécution d’environ 1 ou 6 ms, avec 20 MiB ou 6 MiB utilisée selon les MCU testés.


# 3\. Changement de méthode principale et de répartition du travail {#3.-changement-de-méthode-principale-et-de-répartition-du-travail}

Parmi les 4 méthodes que nous avons testées, si les trois premières présentes des particularités nous poussant à ne pas les utiliser, la quatrième (conversion de pytorch en tflite puis ST Edge AI Developer Cloud) paraît être une option viable pour la réalisation du projet.

Cependant, les demandes matérielles de cette option restent encore une variable à évaluer. En effet, les premières exportations sans aucune optimisation dépassent d’environ 200% la taille de mémoire vive des microcontrôleurs préconisés par Hager Group.

Ainsi, si les Neural Blocks présentés lors des précédentes revues de projet nous semble une option intéressante que l’un d’entre nous continuera d’explorer pour permettre une comparaison des méthodes et une meilleure compréhension du projet à bas niveau, l’essentiel du groupe se concentrera sur l’évaluation de notre nouvelle solution principale, ST Edge AI Developer Cloud. De plus, utiliser une solution fournie par ST afin d’exporter sur un architecture ST nous semble parfaitement cohérent.

Pour les trois membres du groupe continuant sur cette solution principale, la tâche consistera donc à :

* L’exportation du modèle sur un vrai STM32 et des tests d’accuracy et de vitesse.
* Potentiellement des optimisations / entraînements du modèle modifié

# 4\. Conclusion {#4.-conclusion}

Ainsi, après cette prise de recul nécessaire et concluante, notre travail l’an prochain sera réorganisé, afin de proposer au plus vite une solution utilisable au client.

Dans l’hypothèse où la taille de l’export avec ST Edge AI Developer Cloud nous empêche de l’utiliser, le fait de laisser un de nos membres continuer de travailler sur les Neural Blocks nous permettra d’y revenir, et au moins à des fins de compréhension profonde du sujet et de comparaison de performances.

Dans l’état actuel des choses, il nous reste à vérifier avec le client la manière dont on règle le problème de taille de l’export (achat d’une RAM supplémentaire et donc estimation du surcoût ou bien limitation du modèle au prix d’une baisse de précision).


# 5\. Références {#5.-références}

\[1\] [https://pytorch.org/executorch/](https://pytorch.org/executorch/)
\[2\] [https://pytorch.org/executorch/docs/](https://pytorch.org/executorch/docs/]\(https://pytorch.org/executorch/docs/\))
\[3\] [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)
\[4\] [https://www.tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
\[5\] [https://pytorch.org/docs/stable/jit.html](https://pytorch.org/docs/stable/jit.html).
\[6\] [https://stm32ai-cs.st.com/home](https://stm32ai-cs.st.com/home)
\[7\] [https://www.st.com/en/development-tools/stm32cubeide.html](https://www.st.com/en/development-tools/stm32cubeide.html)