import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ==========================================
# 1. CONFIGURATION
# ==========================================
DOSSIER_DATA = "image"
NOM_CSV = "pipe_detection_label.csv" # /!\ VÉRIFIE CE NOM /!\
CHEMIN_CSV = os.path.join(DOSSIER_DATA, NOM_CSV)

COLONNE_CIBLE = "width_m" 

# ==========================================
# 2. CHARGEMENT (VISION + MÉTADONNÉES)
# ==========================================
print(f"--- Extraction des données pour prédire '{COLONNE_CIBLE}' ---")
df = pd.read_csv(CHEMIN_CSV, sep=';')

# --- L'ASTUCE DU ONE-HOT ENCODING ---
colonnes_texte = ['coverage_type', 'shape', 'noisy', 'noise_type', 'pipe_type']
colonnes_a_encoder = [col for col in colonnes_texte if col in df.columns]

if colonnes_a_encoder:
    print(f"-> Encodage des métadonnées du chantier : {colonnes_a_encoder}")
    # On transforme le texte en colonnes mathématiques (0 ou 1)
    df = pd.get_dummies(df, columns=colonnes_a_encoder, drop_first=True)

X = []
y = []
images_ignorees = 0

# On liste toutes les nouvelles colonnes créées par l'encodage
colonnes_ignorees = ['field_file', 'label', COLONNE_CIBLE]
colonnes_meta = [col for col in df.columns if col not in colonnes_ignorees]

for idx, row in df.iterrows():
    # --- LE FILTRE ABSOLU (Que les vrais tuyaux) ---
    try:
        label = int(row['label'])
    except:
        label = 0
        
    if label == 0 or pd.isna(row[COLONNE_CIBLE]):
        images_ignorees += 1
        continue

    largeur = float(row[COLONNE_CIBLE])

    # --- PARTIE VISION : Analyse de l'image ---
    img_name = os.path.join(DOSSIER_DATA, str(row['field_file']))
    data = np.load(img_name)
    img = data[data.files[0]].astype(np.float64)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    h, w, c = img.shape
    features_img = [h, w, w * h]
    
    for canal in range(4):
        canal_data = img[:, :, canal]
        abs_data = np.abs(canal_data)
        
        features_img.append(np.max(canal_data))
        features_img.append(np.min(canal_data))
        features_img.append(np.mean(canal_data))
        features_img.append(np.std(canal_data))
        
        # La taille de la tache magnétique
        seuil = np.max(abs_data) * 0.10
        features_img.append(np.sum(abs_data > seuil))
        features_img.append(np.sum(np.max(abs_data, axis=0) > seuil))
    
    # --- PARTIE TEXTE : Métadonnées du chantier ---
    # On récupère les 0 et les 1 des colonnes encodées
    features_meta = row[colonnes_meta].astype(float).tolist()
    
    # On fusionne la vision et le texte
    features_totales = features_img + features_meta
    
    X.append(features_totales)
    y.append(largeur)

    if len(X) % 100 == 0:
        print(f"   -> {len(X)} vrais tuyaux analysés...")

X = np.array(X)
y = np.array(y)
print(f"-> Tri terminé ! {images_ignorees} images vides (ou sans largeur) jetées à la poubelle.")
print(f"-> L'IA va s'entraîner sur {len(X)} vrais tuyaux avec {X.shape[1]} indices chacun.")

# ==========================================
# 3. SÉPARATION DES DONNÉES (Train / Test)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. LE PIPELINE : STANDARDISATION -> RANDOM FOREST
# ==========================================
print("\n--- Démarrage de l'apprentissage (Régression Multimodale) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modele_reg = RandomForestRegressor(n_estimators=100, random_state=42)
modele_reg.fit(X_train_scaled, y_train)

# ==========================================
# 5. ÉVALUATION DES PERFORMANCES
# ==========================================
print("\n--- Passage de l'examen final sur les 20% de test ---")
y_pred = modele_reg.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) * 100

print(f"\n RÉSULTATS DE LA TÂCHE 2 (Prédiction de {COLONNE_CIBLE}) ")
print(f"Marge d'erreur moyenne (MAE) : {mae:.3f} mètres")
print(f"Score de fiabilité (R²)      : {r2:.2f}%")

if mae < 1.0:
    print("\n INCROYABLE ! L'objectif de la MAE < 1m est atteint ! ")
else:
    print("\n L'erreur est toujours > 1m. Dis-moi le score pour qu'on avise !")

# ==========================================
# 6. SAUVEGARDE DU LIVRABLE
# ==========================================
pipeline_tache2 = {
    'scaler': scaler,
    'modele': modele_reg,
    'colonnes_meta': colonnes_meta # On sauvegarde les noms des colonnes pour le futur
}
joblib.dump(pipeline_tache2, "modele_tache2_reg_final.pkl")
print("\n--- Succès ! Modèle sauvegardé sous 'modele_tache2_reg_final.pkl' ---")