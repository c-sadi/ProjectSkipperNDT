import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns



# 1. CONFIGURATION

DOSSIER_DATA = "image"
CHEMIN_CSV = os.path.join(DOSSIER_DATA, "pipe_detection_label.csv") 
COLONNE_CIBLE = "width_m"
EPOCHS = 30
BATCH_SIZE = 32


# 2. PRÉPARATION DES DONNÉES (PANDAS)

print("--- 1. Lecture et Filtrage du CSV ---")
df = pd.read_csv(CHEMIN_CSV, sep=';')

# Filtrage : 
df = df[df['label'].astype(str) == '1']
df = df.dropna(subset=[COLONNE_CIBLE]).reset_index(drop=True)

# 
colonnes_texte = ['coverage_type', 'shape', 'noisy', 'noise_type', 'pipe_type']
colonnes_a_encoder = [col for col in colonnes_texte if col in df.columns]
if colonnes_a_encoder:
    df = pd.get_dummies(df, columns=colonnes_a_encoder, drop_first=True)


colonnes_ignorees = ['field_file', 'label', COLONNE_CIBLE]
meta_cols = [col for col in df.columns if col not in colonnes_ignorees]

#  Train / Test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


# 3.  (PYTORCH)

class SkipperDataset(Dataset):
    def __init__(self, dataframe, meta_columns):
        self.df = dataframe.reset_index(drop=True)
        self.meta_cols = meta_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- A. L'IMAGE ---
        img_name = os.path.join(DOSSIER_DATA, str(row['field_file']))
        data = np.load(img_name)
        img = data[data.files[0]].astype(np.float32)
        img = np.nan_to_num(img, nan=0.0)
        
        # 
        h, w, _ = img.shape
        
        # Normalisation de l'image (crucial pour le CNN)
        max_val = np.max(np.abs(img))
        if max_val > 0:
            img = img / max_val
            
        # PyTorch veut l'ordre (Canaux, Hauteur, Largeur)
        tensor_img = torch.tensor(img).permute(2, 0, 1)
        
        #  l'image en 128x128 
        tensor_img = F.interpolate(tensor_img.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        
        # -----
        # 
        meta_vals = [float(h), float(w)] + row[self.meta_cols].astype(float).tolist()
        tensor_meta = torch.tensor(meta_vals, dtype=torch.float32)
        
        # --- 
        target = torch.tensor([float(row[COLONNE_CIBLE])], dtype=torch.float32)
        
        return tensor_img, tensor_meta, target

print("--- 2. Création des Dataloaders ---")
train_dataset = SkipperDataset(df_train, meta_cols)
test_dataset = SkipperDataset(df_test, meta_cols)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 4. L'ARCHITECTURE MULTIMODALE (LE CERVEAU)

class MultimodalCNN(nn.Module):
    def __init__(self, num_meta_features):
        super().__init__()
        # Branche VISION (Extrait la forme)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        
        # 
        self.fc_meta = nn.Linear(num_meta_features, 32)
        
        # 
        self.fc_final1 = nn.Linear(1024 + 32, 128)
        self.fc_final2 = nn.Linear(128, 1) 

    def forward(self, img, meta):
        # Passage dans les yeux
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Passage dans les oreilles
        y = F.relu(self.fc_meta(meta))
        
        # Prise de décision globale
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc_final1(z))
        return self.fc_final2(z)

# Initialisation
num_meta = len(meta_cols) + 2 
model = MultimodalCNN(num_meta)
criterion = nn.L1Loss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 5. L'ENTRAÎNEMENT 

print(f"\n--- 3. Début de l'entraînement PyTorch ({EPOCHS} Epochs) ---")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for imgs, metas, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs, metas)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Erreur Moyenne (Train) : {train_loss:.3f} mètres")




print("\n--- 4. Évaluation sur les données invisibles ---")
model.eval()
predictions = []
vraies_valeurs = []

with torch.no_grad():
    for imgs, metas, targets in test_loader:
        outputs = model(imgs, metas)
        predictions.extend(outputs.view(-1).tolist())
        vraies_valeurs.extend(targets.view(-1).tolist())

mae = mean_absolute_error(vraies_valeurs, predictions)
r2 = r2_score(vraies_valeurs, predictions) * 100

print(f"\n RÉSULTATS PYTORCH (TÂCHE 2) ")
print(f"Marge d'erreur moyenne (MAE) : {mae:.3f} mètres")
print(f"Score de fiabilité (R²)      : {r2:.2f}%")

if mae < 1.0:
    print("\n BOUM ! Tu as cassé la barre des 1 mètre avec le Deep Learning ! ")
else:
    print("\n On a une marge à affiner, mais le réseau a vu la lumière !")


    
# graphe



print("\n--- Génération des graphiques Tâche 2 ---")


y_true_np = np.array(vraies_valeurs)
y_pred_np = np.array(predictions)


plt.figure(figsize=(10, 10))
plt.scatter(y_true_np, y_pred_np, alpha=0.6, color='royalblue', s=40, edgecolor='w', label='Prédictions de l\'IA')


max_val = max(np.max(y_true_np), np.max(y_pred_np))
min_val = min(np.min(y_true_np), np.min(y_pred_np))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction 100% Parfaite')

plt.title(f"Régression CNN PyTorch : Prédictions vs Réalité\nR² = {r2:.2f}% | MAE = {mae:.3f} mètres", fontsize=14)
plt.xlabel("Vraie Largeur de la Tache (Mètres)", fontsize=12)
plt.ylabel("Largeur Prédite par le CNN (Mètres)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal') 
plt.tight_layout()
plt.savefig("graphique_tache2_cnn_pred_vs_real.png", dpi=300)
print("-> Graphique Juge de Paix sauvegardé sous 'graphique_tache2_cnn_pred_vs_real.png'")
plt.close() 

# --- GRAPHIQUE 2 
erreurs = y_true_np - y_pred_np
plt.figure(figsize=(10, 6))
sns.histplot(erreurs, kde=True, bins=30, color='purple', edgecolor='w')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zéro Erreur (Parfait)')
plt.title("Distribution des Erreurs (Positif = IA sous-estime, Négatif = IA surestime)", fontsize=13)
plt.xlabel("Erreur en Mètres (Vrai - Prédit)", fontsize=12)
plt.ylabel("Nombre d'images", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("graphique_tache2_cnn_errors_hist.png", dpi=300)
print("-> Histogramme des erreurs sauvegardé sous 'graphique_tache2_cnn_errors_hist.png'")
plt.close()  