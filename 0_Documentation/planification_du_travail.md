# Planification du travail

**Attention:** L’ordre de réalisation de chacune de ces tâches / sous tâches n’est pas forcément linéaire, il faut juste faire attention aux dépendances entre chacunes des tâches et bien suivre les dépendances.

----------------------------- DOCUMENTATION -----------------------------

## 0 - Création d’une documentation de tout ce qui est fait et qui est maintenue à jour au fur et à mesure que le projet avance.

Ce qu’il faut documenter:

- les idées / les algorithmes que l’on met en place
- Les codes informatiques que l’on développe
- Les tests que l’on fait pour vérifier que tout va bien


-------------------------- TRAITEMENT DE MODÈLE --------------------------

## 1 - Conversion (modèles d’IA enregistrés pytorch / tensorflow / etc…) vers langage intermédiaire du type, sous format JSON (Comme ça, simple à charger et facilement compatible, en plus d’être à peu près Human-readable):

Fichier JSON du modèle:
```json
{
    "type": "Model",
    "name": "…",
    "input_shape": (… , ),
    "input_datatype": "…",
    "output_shape": (… , ),
    "output_datatype": "…",
    "blocks": [
        …	// liste de blocs, de type Block
    ]
}
```

Un modèle est composé de différents Blocks, avec la syntaxe suivante:
```json
{
    "type": "Block",
    "name": "…",
    "input_shape": (… , ),
    "input_datatype": "…",
    "output_shape": (… , ),
    "output_datatype": "…",
    "layers": [
        …	// liste de définitions de Tensor, d’opérations directes sur des Tensor, ou de Layers
    ]
}
```

*Note:* L’entrée d’un Block définit automatiquement un Tensor nommé `X`, de la forme et du type précisé, et la sortie d’un Block doit être dans un Tensor nommé `Y`, de la taille et du type précisés.

Représentation d’un Tensor:
```json
{
    "type": "Tensor",
    "name": "…",
    "shape": (… , ),
    "datatype": "…",
    "data": "path_toward_binary_file"	// Par soucis de lisibilité et d’organisation, on va avoir chaque poids du modèle dans un fichier binaire séparé, on pourra les regrouper lors du script final C ou Assembleur
}
```

Représentation d’un Layer:
```json
{
    "type": "Layer",
    "name": "…",		// Le nom doit être parmi le nom des couches supportées
    "tensor_input": "…",	// un nom de tenseur, ou une liste de tenseurs, dépendant du Layer
    "tensor_output": "…"	// un nom de tenseur, ou une liste de tenseurs, dépendant du Layer
}
```

Représentations d’opérations directes sur les tenseurs:

Copie/Lien d’un tenseur vers un autre variable:
```json
{
    "type": "=",
    "tensor_input": "…",
    "tensor_output": "…"
}
```

Opérations arithmétiques de tenseurs:
```json
{
    "type": "+, -, .*, *, ./, %, ^x",
    "tensor_input": "…",	// un nom de tenseur, ou une liste de tenseurs, dépendant de l’opération
    "tensor_output": "…"
}
```

Ce langage intermédiaire permet de représenter les différents blocs de modèles et le flux des données dans le modèle, afin de pouvoir traiter efficacement le modèle, pour pouvoir l’optimiser correctement, et l’exporter en code C efficacement.

### 1.1 - Analyse et compréhension

#### 1.1.1 - Lister toutes les librairies python que l’on va supporter, et analyser / comprendre leurs formats d’exportations des modèles (.h5, .keras, .torch) pour pouvoir les transformer dans ce langage intermédiaire unifié.

Pour l’instant, on va d’abord se concentrer sur Pytorch, et on verra après, si on supporte TensorFlow.

#### 1.1.2 - Lister et documenter tous les formats d’encodage des données utilisés (ex: int8, float64, uint16, int1, …).

*à étudier*

#### 1.1.3 - Lister et documenter toutes les fonctions / couches utilisées (ex: couche dense (tf) = couche linéaire (pytorch), couche de convolution, …).

-> Liste des couches pour langage intermédiaire (voir fichiers `1_Conversion_Intermediate_Language_Model_Architecture/layers.json` / [doc google - Liste des couches pour language intermediaires](https://docs.google.com/document/d/1YU9-MRqxcMLJRASWjCxdkN-StsMlnh8Jp-h2MPxlIJo/edit?usp=sharing) )

#### 1.2 - Designer complètement tout du langage intermédiaire après avoir bien analysé et tout listé.

Fait. Le design juste au-dessus en JSON est complet.

### 1.3 - Écrire un script par format que l’on supporte qui prend en entrée les poids d’un modèle et sa structure et qui rend en sortie le langage intermédiaire.

Il y a un problème, il manque des informations avec seulement un object python model de type nn.Module. Il va falloir soit prendre en entrée directement la structure du modèle, ce qui est trop demandant, et pas flexible du tout, ou bien faire un script qui va directement charger le code python qui contient la définition du modèle, et le compiler pour en obtenir le code intermédiaire.

### 1.4 - Réaliser une banque de tests et vérifier que tout fonctionne bien.

*à étudier*

## 2 - Optimisations sur ce langage intermédiaire

### 2.1 - Mise à l’échelle des opérations linéaires

Ex:

```py
linearLayer(512, 1024)
linearLayer(1024, 256)
```

Peut être optimisé en:

```py
linearLayer(512, d)
linearLayer(d, 256)
```

Avec `d` inférieur à 1024 -> étude de l’évolution des pertes de performances en fonction de `d`.

*à étudier: Comment calculer les matrices de poids des couches optimisées en fonction des anciennes matrices de poids ?*

### 2.2 - Pruning / Suppression de poids

Mettre des poids à zéro

*à étudier: étude des pertes de performances en fonction du nombre et poids que l’on enlève, où est-ce qu’on les enlève ?*

### 2.3 - Pruning / Suppression de couche intermédiaire directement

*à étudier: étude des pertes de performances en fonction du fait qu’on enlève des couches complètes à un modèle*

### 2.4 - Quantization des poids

*à étudier: étude des pertes de performances en fonction de la quantization appliquée au modèle (ex: float64 -> int16, ou float64 -> int4)*


------------------ OPTIMISATION ET EXPORTATION DE CODE ------------------

## 3 - Conversion vers langage intermédiaire préliminaire à C / assembleur du type:

```
(array, var_name, type, data)
(variable, var_name, type, data)
(operation_on_array, (output_var_name_arr, dim_start, dim_end), (input1_var_name, dim_start, dim_end), (input2_var_name, dim_start, dim_end))
…
```

*à étudier plus en profondeur et à décomposer en sous-tâches*

## 4 - Optimisations sur ce langage intermédiaire

utiliser moins de RAM, supprimer les calculs inutiles, détection des calculs redondants, essayer de réduire le plus possible le nombre d’opérations arithmétiques, …

*à étudier plus en profondeur et à décomposer en sous-tâches*

## 5 - Exportation en code C

*à étudier plus en profondeur et à décomposer en sous-tâches*

## 6 - Tests de performances / simulation des modèles sur architectures précises avant déploiement matériel / évaluation des coûts énergétiques des solutions proposées

*à étudier plus en profondeur et à décomposer en sous-tâches*

## (Optionnel mais intéressant) - Exportation en code assembleur (en fonction d’une architecture donnée, registres, opérations supportées, …)

*à étudier plus en profondeur et à décomposer en sous-tâches*
