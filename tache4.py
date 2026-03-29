import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ==========================================
# 1. CONFIGURATION TÂCHE 4
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOSSIER_DATA = "image tache3" # /!\ À adapter selon ton vrai dossier
NOM_CSV = "pipe_detection_label.csv" # /!\ À vérifier aussi
CHEMIN_CSV = os.path.join(DOSSIER_DATA, NOM_CSV)

BATCH_SIZE = 8
EPOCHS = 30
TARGET_SIZE = (128, 128)

# ==========================================
# 2. DATASET AVEC DATA AUGMENTATION INTENSIVE
# ==========================================
class ParallelPipeDataset(Dataset):
    def __init__(self, dataframe, root_dir, target_size=(128,128), augment=False):
        self.df = dataframe
        self.root_dir = root_dir
        self.target_size = target_size
        self.augment = augment # Activation de la triche intelligente !

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.df.iloc[idx]['field_file']))

        try:
            label = int(self.df.iloc[idx]['label'])
        except:
            label = 0

        # Chargement de l'image
        data = np.load(img_name)
        image = data[data.files[0]].astype(np.float32)
        image = np.nan_to_num(image, nan=0.0)

        # Normalisation
        mean = image.mean()
        std = image.std() + 1e-6
        image = (image - mean) / std

        # --- DATA AUGMENTATION (Seulement à l'entraînement) ---
        if self.augment:
            # Effet miroir horizontal 1 fois sur 2
            if random.random() > 0.5:
                image = np.fliplr(image)
            # Effet miroir vertical 1 fois sur 2
            if random.random() > 0.5:
                image = np.flipud(image)

        # Passage au format PyTorch (Canaux, H, W)
        image = np.transpose(image, (2,0,1)).copy() # .copy() est requis après un flip numpy
        image = torch.from_numpy(image).float()

        # Padding pour taille fixe
        c, h, w = image.shape
        pad_h = max(0, self.target_size[0] - h)
        pad_w = max(0, self.target_size[1] - w)
        image = F.pad(image, (0, pad_w, 0, pad_h), "constant", 0)
        image = image[:, :self.target_size[0], :self.target_size[1]]

        return image, torch.tensor(label).long()

# ==========================================
# 3. LE MODÈLE ANTI-SURAPPRENTISSAGE (Dropout)
# ==========================================
class ParallelCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # On rajoute de la profondeur pour capter les "patterns spatiaux subtils"
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2), # Désactive 20% des neurones pour éviter le par cœur

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3), # Désactive 30% des neurones

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4)  # Désactive 40% des neurones
        )

        # Le classificateur final
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Grosse pénalité anti-triche
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==========================================
# 4. PRÉPARATION DES DONNÉES
# ==========================================
print(f"--- Démarrage de la Tâche 4 (Conduites Parallèles) ---")
df = pd.read_csv(CHEMIN_CSV, sep=';')

# Séparation des noms de fichiers avant de créer les datasets
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Le Dataset d'entraînement a 'augment=True' (Data Augmentation)
train_dataset = ParallelPipeDataset(train_df, DOSSIER_DATA, TARGET_SIZE, augment=True)
# Le Dataset de validation est pur (On ne triche pas le jour de l'examen)
val_dataset = ParallelPipeDataset(val_df, DOSSIER_DATA, TARGET_SIZE, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Poids pour déséquilibre
labels = train_df['label'].values
weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

model = ParallelCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# ==========================================
# 5. ENTRAÎNEMENT AVEC SAUVEGARDE DU MEILLEUR MODÈLE
# ==========================================
print(f"--- Entraînement sur {EPOCHS} Epochs avec Data Augmentation et Dropout ---")

best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # --- Évaluation à la fin de chaque Epoch ---
    model.eval()
    all_preds, all_labels_val = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) # On prend le max classique pour l'instant
            
            all_preds.extend(preds.cpu().numpy())
            all_labels_val.extend(labels.numpy())

    acc = accuracy_score(all_labels_val, all_preds)
    f1 = f1_score(all_labels_val, all_preds, zero_division=0)
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss:.4f} | Val Accuracy: {acc:.3f} | Val F1: {f1:.3f}")
    
    # On sauvegarde UNIQUEMENT si on a battu le record de F1-Score !
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "modele_tache4_best.pth")
        print(f"   -> 🏆 Nouveau record F1-Score ! Modèle sauvegardé.")

print(f"\n✅ Fin de l'entraînement. Meilleur F1-Score atteint : {best_f1:.3f} (Objectif > 0.80)") 