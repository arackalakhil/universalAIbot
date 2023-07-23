import json
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import model
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
from nltk_utilis import tokanize, stem, bag_of_words

with open("train.json", "r") as f:
    train = json.load(f)

all_word = []
tags = []
xy = []

for train_item in train["intents"]:
    tag = train_item["tag"]
    tags.append(tag)

    for pattern in train_item["patterns"]:
        w = tokanize(pattern)
        all_word.extend(w)
        xy.append((w, tag))

ignore = ["?", "!", ".", ","]
all_word = [stem(w) for w in all_word if w not in ignore]
all_word = sorted(set(all_word))
tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_word)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

batch_size = 8
hidden_size = 8
input_size = len(all_word)  # Fix this line: len(tag) -> len(all_word)
output_size = len(tags)  # Fix this line: len(x_train[0]) -> len(tags)
learning_rate = 0.001
num_epoch = 1000

data_set = ChatDataset()

train_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.long)  # Convert labels to LongTensor

        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epoch}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')
data ={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "all_words":all_word,
    "tags":tags
}
FILE="data.pth"
torch.save(data,FILE)
print("compltedand file save to{File}")