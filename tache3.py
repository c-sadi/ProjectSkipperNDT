import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOSSIER_DATA = "image"
NOM_CSV = "pipe_detection_label.csv"
CHEMIN_CSV = os.path.join(DOSSIER_DATA, NOM_CSV)

BATCH_SIZE = 8
EPOCHS = 10
TARGET_SIZE = (128, 128)

# ==========================================
# 2. CHARGEMENT CSV
# ==========================================
df = pd.read_csv(CHEMIN_CSV, sep=';')

# ==========================================
# 3. DATASET
# ==========================================
class CurrentIntensityDataset(Dataset):
    def __init__(self, dataframe, root_dir, target_size=(128,128)):
        self.df = dataframe
        self.root_dir = root_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.df.iloc[idx]['field_file']))

        try:
            label = int(self.df.iloc[idx]['label'])
        except:
            label = 0

        data = np.load(img_name)
        image = data[data.files[0]].astype(np.float32)

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        mean = image.mean()
        std = image.std() if image.std() > 1e-6 else 1.0
        image = (image - mean) / std

        image = np.transpose(image, (2,0,1))
        image = torch.from_numpy(image).float()

        c, h, w = image.shape
        pad_h = max(0, self.target_size[0] - h)
        pad_w = max(0, self.target_size[1] - w)
        image = F.pad(image, (0, pad_w, 0, pad_h), "constant", 0)
        image = image[:, :self.target_size[0], :self.target_size[1]]

        return image, torch.tensor(label).long()

# ==========================================
# 4. MODELE CNN
# ==========================================
class CurrentCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(32 * 32 * 32, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 5. PREPARATION DONNEES
# ==========================================
print(f"--- Préparation des données sur {DEVICE} ---")

dataset = CurrentIntensityDataset(df, DOSSIER_DATA, TARGET_SIZE)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==========================================
# 6. PONDERATION DES CLASSES (IMPORTANT)
# ==========================================
labels = df['label'].values
weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

model = CurrentCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# ==========================================
# 7. ENTRAINEMENT
# ==========================================
print("\n--- Démarrage entraînement Tâche 3 ---")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    # ======================================
    # VALIDATION AVEC SEUIL OPTIMISE
    # ======================================
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:,1]

            # seuil pour augmenter le recall
            preds = (probs > 0.4).int().cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"⭐ Accuracy: {acc:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

# ==========================================
# 8. SAUVEGARDE
# ==========================================
torch.save(model.state_dict(), "modele_tache3_intensite.pth")
print("\n✔ Modèle Tâche 3 sauvegardé")