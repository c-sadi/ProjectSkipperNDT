# ProjectSkipperNDT

# 🚁 Projet Data Science : Détection Magnétique de Pipelines (Skipper NDT)


## 📖 Présentation du Projet
Ce projet a été réalisé en partenariat avec **Skipper NDT**, une entreprise française DeepTech spécialisée dans l'inspection et la cartographie 3D des conduites enterrées (pétrole, gaz, eau) à l'aide de capteurs magnétiques embarqués sur des drones.

**L'objectif principal** est d'automatiser l'analyse des cartes de champs magnétiques générées par les drones afin d'isoler les signatures magnétiques des canalisations et d'en évaluer l'état ou les caractéristiques, remplaçant ainsi une analyse visuelle et manuelle fastidieuse.

---

##  Architecture du Projet (Les 4 Tâches)

Le projet est divisé en 4 missions de difficulté croissante, allant du Machine Learning classique au Deep Learning avancé :

### Tâche 1 : Détection de présence de conduites (Classification Binaire)
* **Objectif :** Déterminer si une image d'anomalie magnétique contient ou non une canalisation.
* **Méthode :** Réduction de dimension avec **PCA** (Principal Component Analysis) couplée à un classificateur **K-Nearest Neighbors (KNN)**.
* **Résultat :** Algorithme extrêmement robuste (100% de réussite sur le jeu de test) validant la faisabilité de l'approche algorithmique.

### Tâche 2 : Estimation de la largeur de la conduite (Régression)

### Tâche 3 : Validation de l'intensité du courant de protection (Deep Learning)
* **Objectif :** Classificateur binaire pour vérifier si le courant électrique circulant dans la conduite (pour éviter la corrosion) est suffisant.
* **Méthode :** Création d'un **Convolutional Neural Network (CNN)** avec PyTorch (2 couches convolutives, Batch Normalization).
* **Résultats :** Dépassement des objectifs fixés par l'entreprise avec une **Accuracy > 92%** et un **Recall > 94%**.
* **Visualisation :** *Voir le graphique de convergence dans les livrables.*

### Tâche 4 : Détection de Conduites Parallèles (Transfer Learning & Régularisation)
* **Objectif :** Distinguer une conduite unique de plusieurs conduites parallèles très proches (pattern spatial complexe).
* **Défi :** Très petit volume de données (~300 images) entraînant un fort risque de surapprentissage.
* **Méthode :** Implémentation de techniques anti-overfitting (**Data Augmentation** par rotation/flip, couches de **Dropout**) et utilisation de **Transfer Learning** (ResNet18 modifié pour accepter des images à 4 canaux).
* **Analyse :** Mise en évidence expérimentale des limites d'un CNN classique sur un dataset restreint, justifiant le passage aux architectures pré-entraînées.

---

## 🛠️ Technologies Utilisées
* **Langage :** Python
* **Manipulation de Données :** Pandas, NumPy
* **Machine Learning :** Scikit-Learn (PCA, KNN)
* **Traitement d'Image :** OpenCV
* **Deep Learning :** PyTorch, Torchvision
* **Visualisation :** Matplotlib

---

## 🚀 Comment lancer le projet

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/c-sadi/ProjectSkipperNDT.git](https://github.com/c-sadi/ProjectSkipperNDT.git)
   cd ProjectSkipperNDT
