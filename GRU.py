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

dataset_path = "Audio_Speech_Actors_01-24"
feature_files = glob.glob(os.path.join(dataset_path, "Actor_*", "*.npz"))
max_len = 300

[x_train_torch,
 y_train_torch,
 x_val_torch,
 y_val_torch,
 x_test_torch,
 y_test_torch] = prepare_data(feature_files=feature_files,
                            flatten=False, # Keep temporal info
                            features=['mfcc','rms'], # Feature selection
                            max_len=max_len # Maximum length of data selected - all data will be padded/turnicated to it
                           )
# Our prepare data function return (batch, feature, windows) GRU expects (batch, windows, features)
x_train_torch = x_train_torch.permute(0,2,1)
x_val_torch = x_val_torch.permute(0,2,1)
x_test_torch = x_test_torch.permute(0,2,1)


class GRU(nn.Module):
    def  __init__(self,dropout_rate = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=41,hidden_size=64,num_layers=2,batch_first=True)
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,8)
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self,x):
        output,_ = self.gru(x)
        x = output[:,-1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
model = GRU()
criterion = nn.CrossEntropyLoss()
lr = 0.001 # Learn rate for optimizer
optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)

# Prepare data for training
train_loader = DataLoader(TensorDataset(x_train_torch, y_train_torch), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(x_val_torch,   y_val_torch),   batch_size=32, shuffle=False)

# Training loop
n_epoch = 30
run_loss_vec = []
val_loss_vec = []
best_val_loss = float('inf')
best_epoch = int(0)
for epoch in range(n_epoch):
    # Training
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs + torch.randn_like(inputs) * 0.01
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    run_loss_vec.append(running_loss)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            output = model(inputs)
            val_loss += criterion(output, labels).item()
    val_loss_vec.append(val_loss)
    print(f"Epoch {epoch+1}/{n_epoch} - train loss: {running_loss:.4f}, val loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'gru_best.pth')
        print("Saved")
        best_epoch = epoch

# Plot training loss
plt.figure()
plt.plot(range(1, n_epoch+1), run_loss_vec, label='train')
plt.plot(range(1, n_epoch+1), val_loss_vec, label='val')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
print(f"Best epoch is {best_epoch}")
# Load best model for final evaluation
model.load_state_dict(torch.load('gru_best.pth'))
model.eval()

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(classification_report(all_labels, all_preds, target_names=list(EMOTION_MAP.values())))
print(f'Accuracy of the network on validation dataset: {100 * correct / total:.2f} %')
print("Val label distribution:", np.bincount(y_val_torch.numpy()))
print("Train label distribution:", np.bincount(y_train_torch.numpy()))