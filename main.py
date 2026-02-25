import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import joblib


# 1. CONFIGURATION

DOSSIER_DATA = "image"
NOM_CSV = "pipe_detection_label.csv" 
CHEMIN_CSV = os.path.join(DOSSIER_DATA, NOM_CSV)


# 2. CHARGEMENT (LA MÉTHODE ANTI-CRASH : STATISTIQUES)

print("--- Extraction des signaux magnétiques ... ---")
df = pd.read_csv(CHEMIN_CSV, sep=';')

X = []
y = []
total_images = len(df)

for idx, row in df.iterrows():
    #
    if (idx + 1) % 100 == 0:
        print(f"   -> Analyse de l'image {idx + 1} / {total_images}...")

    
    try:
        label = int(row['label'])
    except:
        label = 0
    y.append(label)

    # 2. Chargement de l'image .npz
    img_name = os.path.join(DOSSIER_DATA, str(row['field_file']))
    data = np.load(img_name)
    img = data[data.files[0]].astype(np.float64)

    # Nettoyage NaN 
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Extraction de 16 statistiques (4 par canal)
    features = []
    for canal in range(4): # On boucle sur les canaux Bx, By, Bz et Norme
        canal_data = img[:, :, canal]
        features.append(np.max(canal_data))  # Le signal le plus fort (le pic du tuyau)
        features.append(np.min(canal_data))  # Le signal le plus faible
        features.append(np.mean(canal_data)) # La moyenne globale
        features.append(np.std(canal_data))  # La force de la variation
    
    # On ajoute ces 16 chiffres à notre liste
    X.append(features)

X = np.array(X)
y = np.array(y)
print(f"-> Terminé avec succès ! Plus aucun problème de RAM.")
print(f"-> Tes images immenses ont été résumées en {X.shape[1]} statistiques chacune.")


# 3. SÉPARATION DES DONNÉES (Train / Test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4.  STANDARDISATION -> PCA -> KNN

print("\n--- Démarrage de l'apprentissage (PCA + KNN) ---")

print("1/3 : Standardisation des données...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("2/3 : Calcul de la PCA (Compression des données)...")
# Comme on n'a que 16 caractéristiques au total, on demande à la PCA d'en garder 10
pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("3/3 : Entraînement du modèle KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)


# 5. ÉVALUATION DES PERFORMANCES

print("\n--- Passage de l'examen final sur les 20% de test ---")
y_pred = knn.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n RÉSULTATS FINAUX | Accuracy: {accuracy:.2f}% | Recall: {recall:.2f}% ")
print(f"\nMatrice de confusion :")
print(f"[{conf_matrix[0,0]} (Vrais Négatifs)  |  {conf_matrix[0,1]} (Fausses Alertes)]")
print(f"[{conf_matrix[1,0]} (Tuyaux Ratés)    |  {conf_matrix[1,1]} (Tuyaux Trouvés)]")


# 6. SAUVEGARDE 

pipeline_livrable = {
    'scaler': scaler,
    'pca': pca,
    'knn': knn
}
joblib.dump(pipeline_livrable, "modele_tache1_knn_final.pkl")
print("\n--- Succès ! Pipeline sauvegardé sous 'modele_tache1_knn_final.pkl' ---")