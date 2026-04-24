import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from data_loader import prepare_data
import pyroomacoustics as pra

# Config 
DATASET_PATH = "Audio_Speech_Actors_01-24"
MAX_LEN      = 300
BATCH_SIZE   = 32
N_EPOCHS     = 70
LR           = 1e-4
DROPOUT      = 0.3
PATIENCE     = 30

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

#  Data
feature_files = glob.glob(os.path.join(DATASET_PATH, "Actor_*", "*.npz"))

[x_train_torch,
 y_train_torch,
 x_val_torch,
 y_val_torch,
 x_test_torch,
 y_test_torch] = prepare_data(feature_files=feature_files,
                              flatten=False,
                              features=['mel3ch'],# [3,128,MAX_LEN]
                              max_len=MAX_LEN,apply_cmvn_flag=True)
print(x_train_torch.shape)
train_loader = DataLoader(TensorDataset(x_train_torch, y_train_torch), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(x_val_torch,   y_val_torch),   batch_size=BATCH_SIZE, shuffle=False)

class FocalLoss(nn.Module):
    def __init__(self,gamma=0.2,weight = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self,logits,labels):
        n_classes = logits.size(1)
        eps = 0.1
        ce = F.cross_entropy(logits,labels,weight=self.weight,label_smoothing=eps,reduction="none")
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma*ce).mean()
    
class TDcnn(nn.Module):
    def __init__(self, num_classes=8, dropout=DROPOUT):
        super().__init__()

        # Initial conv: expand from 3 channels to 64, halve spatial size
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block 1: 64 -> 128
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3, groups=64),   # depthwise
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),                         # pointwise
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 128 -> 256
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 256 -> 512
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)   # GAP: (batch, 512, H, W) -> (batch, 512, 1, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x
    
# Training loop
class_weights = torch.ones(8)
class_weights[0] = 2.0  # neutral is label 0
criterion = FocalLoss(gamma=2.0, weight=class_weights)
model     = TDcnn(dropout=DROPOUT)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,patience=5)

# Training
run_loss_vec      = []
val_loss_vec      = []
best_val_loss     = float('inf')
best_epoch        = 0
epochs_no_improve = 0

for epoch in range(N_EPOCHS):


    
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    run_loss_vec.append(running_loss/len(train_loader.dataset))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            val_loss += criterion(model(inputs), labels).item()
    val_loss_vec.append(val_loss/len(val_loader.dataset))
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{N_EPOCHS} | train: {(running_loss/len(train_loader.dataset)):.4f} | val: {(val_loss/len(val_loader.dataset)):.4f}")

    if val_loss < best_val_loss:
        best_val_loss     = val_loss
        best_epoch        = epoch + 1
        epochs_no_improve = 0
        torch.save(model.state_dict(), '2DCNN_best.pth')
        print("Saved")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"Best epoch: {best_epoch}")

# Loss Curve 
epochs_ran = len(run_loss_vec)
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs_ran + 1), run_loss_vec, label='train')
plt.plot(range(1, epochs_ran + 1), val_loss_vec, label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('2DCNN_loss_curve.png')
plt.show()

# Evaluation
model.load_state_dict(torch.load('2DCNN_best.pth'))
model.eval()

correct    = 0
total      = 0
all_preds  = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

print(classification_report(all_labels, all_preds, target_names=list(EMOTION_MAP.values())))
print(f'Val accuracy: {100 * correct / total:.2f}%')
print(f'Best epoch: {best_epoch}')
print(f"Val label distribution:   {np.bincount(y_val_torch.numpy())}")
print(f"Train label distribution: {np.bincount(y_train_torch.numpy())}")

