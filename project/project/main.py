import torch
import json
import random
import numpy as np
import os
import sys
import io
from model import EnhancedNeuralNet
from nltk_utils import tokenize, get_sentence_embedding, correct_spelling, extract_entities, load_embeddings


class EnhancedChatBot:
    def __init__(self, model_weights='enhanced_model_weights.pth', model_data='enhanced_model_data.json',
                 intents_file='intents.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if files exist
        if not os.path.exists(intents_file):
            raise FileNotFoundError(f"Intents file '{intents_file}' not found.")
        if not os.path.exists(model_data):
            raise FileNotFoundError(f"Model data file '{model_data}' not found.")
        if not os.path.exists(model_weights):
            raise FileNotFoundError(f"Model weights file '{model_weights}' not found.")

        # Load the intents
        try:
            with open(intents_file, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in '{intents_file}'.")
        except Exception as e:
            raise Exception(f"Error loading intents file: {str(e)}")

        # Load the trained model data
        try:
            with open(model_data, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in '{model_data}'.")
        except Exception as e:
            raise Exception(f"Error loading model data: {str(e)}")

        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.embedding_dim = model_data.get('embedding_dim', 100)
        self.tags = model_data['tags']

        # Load word embeddings
        print("Loading word embeddings...")
        load_embeddings()

        # Initialize the enhanced model
        self.model = EnhancedNeuralNet(
            input_size=None,  # Not needed for embedding-based model
            hidden_size=self.hidden_size,
            num_classes=self.output_size,
            embedding_dim=self.embedding_dim
        ).to(self.device)

        # Load model weights with weights_only=True to avoid warning
        try:
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device, weights_only=True))
            self.model.eval()
            print(f"Model loaded from {model_weights}")
        except Exception as e:
            raise Exception(f"Error loading model weights: {str(e)}")

        # Initialize conversation context
        self.context = {
            'current_city': None,
            'entities': {
                'locations': [],
                'dates': [],
                'landmarks': [],
                'travel_info': []
            },
            'last_intent': None,
            'conversation_history': []
        }

        print(f"Enhanced ChatBot initialized on {self.device}")

    def preprocess_input(self, user_input):
        """
        Preprocess the user input by tokenizing, correcting spelling, and extracting entities.
        """
        # Tokenize the input text
        tokens = tokenize(user_input)

        # Correct spelling in tokens
        corrected_tokens = [correct_spelling(token) for token in tokens]

        # Extract entities like locations, dates, landmarks, and other relevant information
        entities = extract_entities(user_input)

        # Update context with the entities found in the input
        self.context['entities'] = entities

        return corrected_tokens, entities

    # Modify the get_response method in EnhancedChatBot class

    def get_response(self, user_input):
        """
        Get a response to the user input.
        1. Preprocess input
        2. Feed the input to the model for prediction
        3. Generate a response based on the model's output
        """
        # Preprocess the input
        tokens, entities = self.preprocess_input(user_input)

        # Check if this is a follow-up question about the same context
        is_followup = False
        current_context = None

        # Common follow-up phrases that don't mention the city
        followup_phrases = [
            "what about", "tell me about", "how about", "and", "what's", "how's",
            "any tips", "more info", "more information", "more about"
        ]

        # Check if this is likely a follow-up question
        lower_input = user_input.lower()
        for phrase in followup_phrases:
            if phrase in lower_input:
                is_followup = True
                break

        # If we have locations in context and this seems like a follow-up
        if is_followup and self.context['entities']['locations']:
            current_context = self.context['entities']['locations'][0]
        else:
            # Update context with new entities
            if entities['locations']:
                current_context = entities['locations'][0]
                self.context['current_city'] = current_context

        # Convert text directly to embeddings
        input_embedding = get_sentence_embedding(user_input)

        # Convert the embeddings to tensor and move to the appropriate device (CPU or GPU)
        input_tensor = torch.tensor(input_embedding, dtype=torch.float32).to(self.device)

        # Add a batch dimension if it's missing
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # Get model output (prediction)
        with torch.no_grad():
            output = self.model(input_tensor)

        # Check output dimensions and handle different cases
        if output.dim() == 1:
            # If output is 1D, just get the max value directly
            predicted_class = torch.argmax(output).item()
        else:
            # If output has batch dimension, use dim=1
            _, predicted_class = torch.max(output, dim=1)
            predicted_class = predicted_class.item()

        tag = self.tags[predicted_class]

        # If this is a topic-specific tag (like food, weather, etc.)
        # and we have a city context, try to find a city-specific response
        if current_context and "_" in tag and tag.split("_")[1] in ["food", "weather", "landmarks", "hidden_gems"]:
            topic = tag.split("_")[1]
            city_specific_tag = f"{current_context.lower()}_{topic}"

            # Check if we have a tag for this city and topic
            if city_specific_tag in self.tags:
                tag = city_specific_tag
            elif f"{current_context.lower()}_landmarks" in self.tags and topic != "landmarks":
                # Fallback to general city tag if we can't find specific topic
                tag = f"{current_context.lower()}_landmarks"

        # Store the last intent for future reference
        self.context['last_intent'] = tag

        # Retrieve the appropriate response from intents based on predicted tag
        response = self.get_intent_response(tag)

        # Add to conversation history
        self.context['conversation_history'].append({
            'user_input': user_input,
            'response': response,
            'tag': tag
        })

        return response

    def get_intent_response(self, tag):
        """
        Get the response based on the intent (tag).
        """
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                # Randomly select a response from the intent
                response = random.choice(intent['responses'])
                if not isinstance(response, str):
                    print(f"Error: Response is not a string! {response}")
                    return "Sorry, I didn't understand that."
                return response

        return "Sorry, I didn't understand that."

    def reset_context(self):
        """
        Reset the context after a conversation ends.
        """
        self.context = {
            'current_city': None,
            'entities': {
                'locations': [],
                'dates': [],
                'landmarks': [],
                'travel_info': []
            },
            'last_intent': None,
            'conversation_history': []
        }
        print("Conversation context has been reset.")


def chat():
    """
    Run an interactive chat session with the travel chatbot.
    """
    # Set the correct encoding for stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("Starting the TravelBot chatbot...")

    try:
        chatbot = EnhancedChatBot()

        print("\n" + "=" * 50)
        print("Welcome to TravelBot!")
        print("I can help you with information about Paris, London, Tokyo, and more.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("=" * 50 + "\n")

        # Main conversation loop
        while True:
            # Get user input
            user_input = input("You: ")

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nTravelBot: Goodbye! Safe travels and have a wonderful journey!")
                break

            # Get chatbot response
            try:
                response = chatbot.get_response(user_input)
                print(f"\nTravelBot: {response}\n")
            except Exception as e:
                print(f"\nTravelBot: I'm having trouble processing that. Could you try again?\n")
                print(f"Error: {str(e)}")

    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return


if __name__ == "__main__":
    chat()