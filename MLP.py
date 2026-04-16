import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os
from tqdm import tqdm
from data_loader import load_features
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

# Load data
records = load_features(feature_files,flatten=True,return_gender = False)

# Train/validation/test split
# We will seperate test dataset by actors to avoid leakage
# Test set
 
TEST_ACTORS = {21,22,23,24} # Hardcoding test set to avoid touching it during training
test_data = [r for r in records if r['actor_id'] in TEST_ACTORS]
x_test = np.array([r['features'] for r in test_data])
y_test = np.array([r['label'] for r in test_data])

# Train/valid set
train_valid_data = [r for r in records if r['actor_id'] not in TEST_ACTORS]

# Train/val split
# We'll start by defining a male set and female set so we'll have an equal split between train and val
uniq_actor_id = np.unique([r['actor_id'] for r in train_valid_data])
np.random.seed(6283)

male_set = [a for a in uniq_actor_id if a%2 == 1]
female_set = [a for a in uniq_actor_id if a%2 == 0]

# Shuffle
np.random.shuffle(male_set)
np.random.shuffle(female_set)

# Set
val_actors = set(male_set[:2] + female_set[:2]) # 2M + 2F actors
train_actors = set(male_set[2:]+female_set[2:]) # 8M + 8F actors

train_data = [a for a in train_valid_data if a['actor_id'] not in val_actors]
x_train = np.array([r['features'] for r in train_data])
y_train = np.array([r['label'] for r in train_data])

val_data = [a for a in train_valid_data if a['actor_id'] in val_actors]
x_val = np.array([r['features'] for r in val_data])
y_val = np.array([r['label'] for r in val_data])


# Normalize the data
# Define standard normalization (sklearn does it better. but still ☻☺☻)
def fit_normalizer(X):
    mean = X.mean(axis = 0)
    std = X.std(axis=0)
    return mean,std
def normalize(X,mean,std):
    return (X-mean)/(std+1e-8)

# Normalize
# Fit - We normalize using train data to avoid leakage

train_mean, train_std = fit_normalizer(x_train)

x_train_norm = normalize(x_train,train_mean,train_std)
x_val_norm = normalize(x_val,train_mean,train_std)
x_test_norm = normalize(x_test,train_mean,train_std)

# To torch tensors
# Test
x_test_torch = torch.tensor(x_test_norm, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

# valid
x_val_torch = torch.tensor(x_val_norm,dtype=torch.float32)
y_val_torch = torch.tensor(y_val,dtype=torch.long)

# Train
x_train_torch = torch.tensor(x_train_norm,dtype=torch.float32)
y_train_torch = torch.tensor(y_train,dtype=torch.long)

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
print("Val label distribution:", np.bincount(y_val))
print("Train label distribution:", np.bincount(y_train))