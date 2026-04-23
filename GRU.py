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

# Config 
DATASET_PATH = "Audio_Speech_Actors_01-24"
MAX_LEN      = 300
BATCH_SIZE   = 32
N_EPOCHS     = 200
LR           = 0.005
DROPOUT      = 0.3
NOISE_STD    = 0.01
PATIENCE     = 50

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
                              features=['mfcc', 'rms'],
                              max_len=MAX_LEN)

# prepare_data returns (batch, features, time), GRU expects (batch, time, features)
x_train_torch = x_train_torch.permute(0, 2, 1)
x_val_torch   = x_val_torch.permute(0, 2, 1)
x_test_torch  = x_test_torch.permute(0, 2, 1)

print(f"Train: {x_train_torch.shape} | Val: {x_val_torch.shape} | Test: {x_test_torch.shape}")

# Model 
class GRU(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.gru     = nn.GRU(input_size=41, hidden_size=64, num_layers=2, batch_first=True)
        self.fc1     = nn.Linear(64, 32)
        self.fc2     = nn.Linear(32, 8)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output, _ = self.gru(x)
        x = output[:, -1, :]       # last time step: (batch, hidden_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)            # raw logits
        return x

model     = GRU(dropout_rate=DROPOUT)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.4)

train_loader = DataLoader(TensorDataset(x_train_torch, y_train_torch), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(x_val_torch,   y_val_torch),   batch_size=BATCH_SIZE, shuffle=False)

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
        inputs = inputs + torch.randn_like(inputs) * NOISE_STD
        output = model(inputs)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    run_loss_vec.append(running_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            val_loss += criterion(model(inputs), labels).item()
    val_loss_vec.append(val_loss)
    scheduler.step()
    print(f"Epoch {epoch+1}/{N_EPOCHS} | train: {running_loss:.4f} | val: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss     = val_loss
        best_epoch        = epoch + 1
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'gru_best.pth')
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
plt.savefig('gru_loss_curve.png')
plt.show()

# Evaluation
model.load_state_dict(torch.load('gru_best.pth'))
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