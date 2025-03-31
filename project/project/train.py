import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from nltk_utils import tokenize, get_sentence_embedding, load_embeddings
from model import EnhancedNeuralNet
import time
import os


class EmbeddingDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.embeddings = []

        # Convert all sentences to embeddings
        for sentence in sentences:
            embedding = get_sentence_embedding(sentence)
            self.embeddings.append(embedding)

        self.embeddings = np.array(self.embeddings)

        # Convert labels to numpy array of integers (ensure they are integers)
        self.labels = np.array(self.labels, dtype=np.int64)  # This ensures labels are integers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return embeddings as float32, labels as long (integer)
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def train_enhanced_chatbot(intents_file='intents.json',
                           model_weights_file='enhanced_model_weights.pth',
                           model_data_file='enhanced_model_data.json',
                           plot_results=True):
    """Train the enhanced chatbot model with word embeddings"""
    # Ensure embeddings are loaded
    load_embeddings()

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

    # Process all patterns and tags
    all_sentences = []
    all_tags = []
    tags = []

    for intent in intents['intents']:
        tag = intent['tag']
        if tag not in tags:
            tags.append(tag)

        for pattern in intent['patterns']:
            all_sentences.append(pattern)
            all_tags.append(tags.index(tag))

    # Verify we have data
    if not all_sentences or not all_tags:
        raise ValueError("No training data found in intents file. Make sure it contains patterns.")

    # Convert to numpy arrays
    all_tags = np.array(all_tags)

    # Enhanced hyperparameters
    batch_size = 8
    hidden_size = 128  # Increased for better capacity
    output_size = len(tags)
    embedding_dim = 100  # GloVe embedding dimension
    learning_rate = 0.001
    num_epochs = 1000
    dropout_rate = 0.3

    # Create dataset and dataloader
    print("Creating dataset with embeddings...")
    dataset = EmbeddingDataset(all_sentences, all_tags)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = EnhancedNeuralNet(
        input_size=None,  # Not needed for embedding model
        hidden_size=hidden_size,
        num_classes=output_size,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    # Training loop
    print("Starting training...")
    losses = []
    start_time = time.time()

    try:
        for epoch in range(num_epochs):
            model.train()  # Ensure model is in training mode
            running_loss = 0.0

            for embeddings, labels in train_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()  # This computes the gradients
                optimizer.step()  # Updates the weights

            # Calculate average loss
            epoch_loss = running_loss / len(train_loader)
            losses.append(epoch_loss)

            # Update learning rate
            scheduler.step(epoch_loss)

            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

                # Evaluate on some examples
                if (epoch + 1) % 200 == 0:
                    model.eval()
                    with torch.no_grad():
                        # Test on a few examples
                        test_sentences = [
                            "What can I see in London?",
                            "Tell me about Paris weather",
                            "What food should I try in Tokyo?"
                        ]

                        print("\nTesting on examples:")
                        for sentence in test_sentences:
                            embedding = get_sentence_embedding(sentence)
                            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)

                            output = model(embedding_tensor)

                            # Ensure the output is of shape [batch_size, num_classes] (e.g., [1, num_classes] for single sentence)
                            output = output.squeeze(0)  # Remove the first dimension (batch size 1)

                            # Get the predicted class index
                            _, predicted = torch.max(output, dim=0)

                            tag_idx = predicted.item()
                            tag = tags[tag_idx]

                            print(f"Sentence: '{sentence}'")
                            print(f"Predicted tag: {tag}")
                            print("-" * 30)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate final model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)

            # Ensure labels are Long type (required for CrossEntropyLoss)
            labels = labels.to(device, dtype=torch.long)  # Convert labels to Long if needed

            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Final training accuracy: {accuracy:.2f}%")

    # Create directory for model if it doesn't exist
    model_dir = os.path.dirname(model_weights_file)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save model
    try:
        torch.save(model.state_dict(), model_weights_file)
        print(f"Model saved to {model_weights_file}")
    except Exception as e:
        print(f"Error saving model weights: {str(e)}")

    # Save model data
    model_data = {
        "hidden_size": hidden_size,
        "output_size": output_size,
        "embedding_dim": embedding_dim,
        "dropout_rate": dropout_rate,
        "tags": tags,
        "training_accuracy": accuracy,
        "training_time": training_time
    }

    try:
        with open(model_data_file, "w") as f:
            json.dump(model_data, f)
        print(f"Model data saved to {model_data_file}")
    except Exception as e:
        print(f"Error saving model data: {str(e)}")

    # Plot training progress
    if plot_results:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)

            # Save plot
            plot_file = 'enhanced_training_loss.png'
            plt.savefig(plot_file)
            plt.show()
            print(f"Training plot saved to {plot_file}")
        except Exception as e:
            print(f"Error creating plot: {str(e)}")

    return model, accuracy


if __name__ == "__main__":
    print("Starting Enhanced Chatbot Training...")

    try:
        train_enhanced_chatbot()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have a valid intents.json file in the current directory.")
    except Exception as e:
        print(f"Training failed: {str(e)}")
