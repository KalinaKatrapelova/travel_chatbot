import json
from project.nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from multiprocessing import freeze_support


class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def train_chatbot():
    # Load and process data
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    # Process all words and tags
    all_words = []
    tags = []
    xy = []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ["?", "!", ".", ","]
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Create training data
    x_train = []
    y_train = []
    for (pattern_sen, tag) in xy:
        bag = bag_of_words(pattern_sen, all_words)
        x_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(all_words)
    learning_rate = 0.001
    num_epochs = 1000

    # Create dataset and dataloader
    dataset = ChatDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Training started...")
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}")

    print(f"Final loss = {loss.item():.4f}")

    # Save model and data separately
    print("Saving model...")

    # Save model weights with weights_only=True
    torch.save(model.state_dict(), "model_weights.pth")

    # Save model data
    model_data = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    with open("model_data.json", "w") as f:
        json.dump(model_data, f)

    print("Training complete. Model and data saved!")


if __name__ == '__main__':
    freeze_support()
    train_chatbot()