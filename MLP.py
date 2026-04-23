import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os
from tqdm import tqdm
from data_loader import prepare_data
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class Net(nn.Module): 

    def __init__(self, dropout_coeff = 0.3):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(122,256) # Input is 122 feature
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,8)
        self.dropout = nn.Dropout(dropout_coeff)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x) # Output (no relu - raw logits for CrossEntropyLoss)
        return x

model = Net()
criterion = nn.CrossEntropyLoss()
momentum = 0.9 # Momentum for SGD optimizer
lr = 0.001 # Learn rate for SGD
optimizer = optim.Adam(model.parameters(),lr = lr)  # Stochastic Gradient descent optimizer

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

# Load, split, normalize and convert to tensors
x_train_torch, y_train_torch, x_val_torch, y_val_torch, x_test_torch, y_test_torch = prepare_data(
    feature_files,
    flatten=True,
    features=['mfcc', 'delta', 'delta2', 'zcr', 'rms'],
    test_actors={21, 22, 23, 24},
    val_per_gender=2,
    seed=6283
)

# Batching
train_dataset = TensorDataset(x_train_torch, y_train_torch)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
n_epochs = 100 # Number of Epochs
model.train()
loss_vector = []
for epoch in range(n_epochs):

    running_loss = 0.0
    for input, label in train_loader:

        optimizer.zero_grad()

        outputs = model(input)
        loss = criterion(outputs,label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader.dataset):.4f}")
    loss_vector.append(running_loss/len(train_loader.dataset))
    if (epoch+1) in {1,50,100}:
        print(f"Running loss of epoch {epoch+1} is:{running_loss}")
print('Finished Training')

plt.figure()
plt.plot(np.arange(1, n_epochs+1), loss_vector)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

torch.save(model.state_dict(), 'mlp_weights.pth')

# Validation loop

correct = 0 
total = 0 

val_dataset = TensorDataset(x_val_torch, y_val_torch)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for input, label in val_loader:

        outputs = model(input)
        _,predicted = torch.max(outputs,1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(label.numpy())
        total += label.size(0)
        correct+= (predicted==label).sum().item()

print(classification_report(all_labels, all_preds, target_names=list(EMOTION_MAP.values())))
print(f'Accuracy of the network on validation dataset: {100 * correct / total:.2f} %')
print("Val label distribution:", np.bincount(y_val_torch.numpy()))
print("Train label distribution:", np.bincount(y_train_torch.numpy()))