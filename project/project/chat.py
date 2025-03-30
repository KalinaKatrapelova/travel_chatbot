import torch
import json
import random
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


class SimpleChatBot:
    def __init__(self, model_weights='model_weights.pth', model_data='model_data.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the intents
        with open('intents.json', 'r') as f:
            self.intents = json.load(f)

        # Load the trained model data
        with open(model_data, 'r') as f:
            model_data = json.load(f)

        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.all_words = model_data['all_words']
        self.tags = model_data['tags']

        # Get dropout rate if available in model data (for backward compatibility)
        dropout_rate = model_data.get('dropout_rate', 0.2)

        # Initialize the model with enhanced architecture
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size, dropout_rate).to(self.device)
        # Fixed the warning by setting weights_only=True
        self.model.load_state_dict(torch.load(model_weights, map_location=self.device, weights_only=True))
        self.model.eval()  # Set the model to evaluation mode (important for dropout)

    def get_response(self, sentence, threshold=0.75):
        # Tokenize and create bag of words
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)

        # Set model to eval mode and disable gradient calculation for inference
        self.model.eval()
        with torch.no_grad():
            # Get prediction
            output = self.model(X)
            _, predicted = torch.max(output, dim=1)
            tag = self.tags[predicted.item()]

            # Calculate probability
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]

            # Only respond if probability is high enough
            if prob.item() > threshold:
                for intent in self.intents['intents']:
                    if tag == intent["tag"]:
                        return random.choice(intent['responses'])

        return "I'm not sure I understand. You can ask me about landmarks, food, weather, and local attractions in Paris, London, or Tokyo!"


def chat():
    chatbot = SimpleChatBot()
    bot_name = "TravelBot"

    # Initial greeting with clear formatting
    print(f"{bot_name}: Hello! I'm your travel assistant. I can help you with:")
    print("• Landmarks and attractions in Paris, London, and Tokyo")
    print("• Local food recommendations and dining tips")
    print("• Weather information and best times to visit")
    print("• Cultural experiences and local customs")
    print("\nHow can I help you plan your journey? (Type 'quit' to exit)")

    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        response = chatbot.get_response(sentence)
        print(f"{bot_name}: {response}")


if __name__ == '__main__':
    chat()