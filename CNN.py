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

# I am keeping temporal information so inputs needed to be prepared to the same length
# Method chosen for this is zero padding/tail cutting since most recordings are of similar length
# First visualise the distribution of number of windows to decide on max_len
window_counts = []
for filepath in feature_files:
    data = np.load(filepath)
    window_counts.append(data['mfcc'].shape[1]) # mfcc shape is (40, W)

window_counts = np.array(window_counts)
print(f"Min windows: {window_counts.min()}, Max: {window_counts.max()}, Mean: {window_counts.mean():.1f}, Std: {window_counts.std():.1f}")

plt.figure()
plt.hist(window_counts, bins=30)
plt.xlabel('Number of windows')
plt.ylabel('Count')
plt.title('Distribution of window counts across audio files')
plt.axvline(window_counts.mean(), color='r', linestyle='--', label=f'Mean ({window_counts.mean():.0f})')
plt.legend()
plt.show()

max_len = 300  

# Load data
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

# Define CNN model
class CNN(nn.Module):
    def __init__(self,dropout_rate = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=41,out_channels=32,kernel_size=5) # conv layer returns T - (kernel size - 1)
        self.pool = nn.MaxPool1d(2,2) # Max pooling keeps max value in a 2 window with 2 stride (halves the size)
        self.conv2 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=5)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,8)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(32) # Batch normalization normalized batch
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.mean(dim = -1) # Global Average Pooling to avoid flattening (batch size,256,T) to (batch size,256)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
model = CNN()
criterion = nn.CrossEntropyLoss()
lr = 0.001 # Learn rate for optimizer
optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)

# Prepare data for training
train_loader = DataLoader(TensorDataset(x_train_torch, y_train_torch), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(x_val_torch,   y_val_torch),   batch_size=32, shuffle=False)

# Training loop
n_epoch = 100
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
        torch.save(model.state_dict(), 'cnn_best.pth')
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
model.load_state_dict(torch.load('cnn_best.pth'))
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
 
    
