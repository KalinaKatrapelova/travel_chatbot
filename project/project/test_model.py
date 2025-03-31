import torch
import json
from model import EnhancedNeuralNet
from nltk_utils import bag_of_words, tokenize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def test_model(model_weights='model_weights.pth', model_data='model_data.json', intents_file='intents.json'):
    # Load model data
    with open(model_data, 'r') as f:
        model_data = json.load(f)

    # Load intents for test patterns
    with open(intents_file, 'r') as f:
        intents = json.load(f)

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = model_data['input_size']
    hidden_size = model_data['hidden_size']
    output_size = model_data['output_size']
    all_words = model_data['all_words']
    tags = model_data['tags']
    dropout_rate = model_data.get('dropout_rate', 0.2)

    # Initialize model
    model = EnhancedNeuralNet(input_size, hidden_size, output_size, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()

    # Prepare test data - use patterns from intents
    X_test = []
    y_test = []
    original_text = []

    for intent in intents['intents']:
        tag = intent['tag']
        tag_idx = tags.index(tag)

        for pattern in intent['patterns']:
            tokenized_pattern = tokenize(pattern)
            bag = bag_of_words(tokenized_pattern, all_words)
            X_test.append(bag)
            y_test.append(tag_idx)
            original_text.append(pattern)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Run predictions
    y_pred = []
    probabilities = []

    with torch.no_grad():
        for i in range(len(X_test)):
            x = X_test[i]
            x = x.reshape(1, x.shape[0])
            x = torch.from_numpy(x).to(device)

            output = model(x)
            _, predicted = torch.max(output, dim=1)
            probs = torch.softmax(output, dim=1)

            y_pred.append(predicted.item())
            probabilities.append(probs[0][predicted.item()].item())

    # Calculate accuracy and print report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=tags)
    print(report)

    # Print some example predictions
    print("\nSample Predictions:")
    for i in range(min(10, len(original_text))):
        print(f"Text: '{original_text[i]}'")
        print(f"Actual Tag: {tags[y_test[i]]}")
        print(f"Predicted Tag: {tags[y_pred[i]]}")
        print(f"Confidence: {probabilities[i] * 100:.2f}%")
        print("-" * 50)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(tags))
    plt.xticks(tick_marks, tags, rotation=45)
    plt.yticks(tick_marks, tags)

    # Add labels
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == '__main__':
    # Test original model
    print("Testing original model:")
    test_model()

    # Test improved model (if it exists)
    try:
        print("\nTesting improved model:")
        test_model('enhanced_model_weights.pth', 'enhanced_model_data.json')
    except FileNotFoundError:
        print("\nEnhanced model not found. Please train it first.")