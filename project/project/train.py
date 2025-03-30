import json
from nltk_utils import tokenize, stem, bag_of_words
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


def train_chatbot(intents_file='intents.json', model_weights_file='model_weights.pth',
                  model_data_file='model_data.json'):
    # Load and process data
    with open(intents_file, 'r') as f:
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

    # Improved Hyperparameters
    batch_size = 8
    hidden_size = 64  # Increased from 8 to 64
    output_size = len(tags)
    input_size = len(all_words)
    learning_rate = 0.001
    num_epochs = 1000
    dropout_rate = 0.2  # Add dropout rate

    # Create dataset and dataloader
    dataset = ChatDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model initialization with enhanced architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size, dropout_rate).to(device)

    # Use weight decay in optimizer for additional regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler to reduce learning rate over time
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    # Training loop with validation
    print("Training started...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Update learning rate
        scheduler.step(epoch_loss)

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1}/{num_epochs}, loss = {epoch_loss:.4f}")

    print(f"Final loss = {epoch_loss:.4f}")

    # Save model and data separately
    print("Saving model...")

    # Save model weights
    torch.save(model.state_dict(), model_weights_file)

    # Save model data
    model_data = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "dropout_rate": dropout_rate,
        "all_words": all_words,
        "tags": tags
    }

    with open(model_data_file, "w") as f:
        json.dump(model_data, f)

    print(f"Training complete. Model and data saved to {model_weights_file} and {model_data_file}!")
    return model


if __name__ == '__main__':
    freeze_support()
    train_chatbot()