import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from nltk_utils import tokenize, stem, bag_of_words, get_sentence_embedding, load_embeddings
from torch.utils.data import Dataset, DataLoader
from model import EnhancedNeuralNet
import torch.nn as nn
import time
import os


# Define EmbeddingDataset class here instead of importing from train
class EmbeddingDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

        # Ensure inputs are numpy arrays
        if not isinstance(self.sentences, np.ndarray):
            self.sentences = np.array(self.sentences)
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return embeddings as float32, labels as long (integer)
        return torch.tensor(self.sentences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def experiment_architectures(intents_file='intents.json'):
    # Check if intents file exists
    if not os.path.exists(intents_file):
        raise FileNotFoundError(f"Intents file '{intents_file}' not found. Please make sure it exists.")

    # Load intents
    try:
        with open(intents_file, 'r') as f:
            intents = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in '{intents_file}'. Please check the file.")
    except Exception as e:
        raise Exception(f"Error loading intents file: {str(e)}")

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

    # Create configurations to test
    configurations = [
        {"name": "Baseline", "hidden_size": 8, "dropout": 0.0, "leaky": False},
        {"name": "Larger Hidden", "hidden_size": 64, "dropout": 0.0, "leaky": False},
        {"name": "With Dropout", "hidden_size": 64, "dropout": 0.2, "leaky": False},
        {"name": "With LeakyReLU", "hidden_size": 64, "dropout": 0.0, "leaky": True},
        {"name": "Medium with Regularization", "hidden_size": 32, "dropout": 0.2, "leaky": True},
        {"name": "Full Improved", "hidden_size": 64, "dropout": 0.2, "leaky": True}
    ]

    results = []

    # Train each configuration
    for config in configurations:
        print(f"\nTraining {config['name']} configuration:")
        print(f"Hidden Size: {config['hidden_size']}, Dropout: {config['dropout']}, LeakyReLU: {config['leaky']}")

        # Parameters
        batch_size = 8
        hidden_size = config['hidden_size']
        output_size = len(tags)
        input_size = len(all_words)
        learning_rate = 0.001
        num_epochs = 1000
        dropout_rate = config['dropout']

        # Create dataset and dataloader
        dataset = EmbeddingDataset(x_train, y_train)
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Model initialization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnhancedNeuralNet(input_size=input_size, hidden_size=hidden_size,
                                  num_classes=output_size, embedding_dim=input_size,
                                  dropout_rate=dropout_rate).to(device)

        # If we're not using leaky ReLU, replace it with regular ReLU
        if not config['leaky']:
            model.leaky_relu = torch.nn.ReLU()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Track training progress
        losses = []
        train_start_time = time.time()

        # Training loop
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
                losses.append(loss.item())
                print(f"epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}")

        train_time = time.time() - train_start_time

        # Evaluation on training data
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(device)
                outputs = model(words)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        # Save results
        results.append({
            "name": config['name'],
            "hidden_size": config['hidden_size'],
            "dropout": config['dropout'],
            "leaky": config['leaky'],
            "final_loss": loss.item(),
            "training_time": train_time,
            "train_accuracy": accuracy,
            "losses": losses
        })

        print(f"Configuration: {config['name']}")
        print(f"Final loss: {loss.item():.4f}")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Training accuracy: {accuracy * 100:.2f}%")

        # Save the best model (full improved)
        if config['name'] == "Full Improved":
            # Create directory for model if it doesn't exist
            model_dir = os.path.dirname("enhanced_model_weights.pth")
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)

            torch.save(model.state_dict(), "enhanced_model_weights.pth")
            model_data = {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size,
                "dropout_rate": dropout_rate,
                "all_words": all_words,
                "tags": tags
            }
            with open("enhanced_model_data.json", "w") as f:
                json.dump(model_data, f)

    # Compare results
    plt.figure(figsize=(12, 6))
    for result in results:
        plt.plot(range(1, len(result['losses']) + 1), result['losses'], label=result['name'])

    plt.xlabel('Hundred Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.savefig('model_comparison.png')
    plt.show()

    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    accuracies = [result['train_accuracy'] * 100 for result in results]
    plt.bar([result['name'] for result in results], accuracies)
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.show()

    return results


if __name__ == '__main__':
    experiment_architectures()