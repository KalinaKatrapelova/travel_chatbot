import torch
import json
import random
import numpy as np
import os
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
            with open(intents_file, 'r') as f:
                self.intents = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in '{intents_file}'.")
        except Exception as e:
            raise Exception(f"Error loading intents file: {str(e)}")

        # Load the trained model data
        try:
            with open(model_data, 'r') as f:
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

        # Load model weights
        try:
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
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

    def get_response(self, user_input):
        """
        Get a response to the user input.
        1. Preprocess input
        2. Feed the input to the model for prediction
        3. Generate a response based on the model's output
        """
        # Preprocess the input
        tokens, entities = self.preprocess_input(user_input)

        # Convert tokens to embeddings
        input_embedding = get_sentence_embedding(tokens)

        # Convert the embeddings to tensor and move to the appropriate device (CPU or GPU)
        input_tensor = torch.tensor(input_embedding).to(self.device)

        # Get model output (prediction)
        with torch.no_grad():
            output = self.model(input_tensor)

        # Get the predicted class (tag)
        _, predicted_class = torch.max(output, dim=1)
        tag = self.tags[predicted_class.item()]

        # Retrieve the appropriate response from intents based on predicted tag
        response = self.get_intent_response(tag)

        # Update conversation context
        self.update_conversation_context(user_input, response)

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
                    print(f"Error: Response is not a string! {response}")  # Debugging line
                    return "Sorry, I didn't understand that."
                return response

        return "Sorry, I didn't understand that."

    def update_conversation_context(self, user_input, response):
        """
        Update the conversation context (e.g., current city, entities, etc.) based on user input and response.
        """
        # Ensure response is a string before using .lower()
        if isinstance(response, str) and 'location' in response.lower():
            self.context['current_city'] = self.extract_location_from_response(response)

        # Add to conversation history
        self.context['conversation_history'].append({
            'user_input': user_input,
            'response': response
        })

        return self.context

    def extract_location_from_response(self, response):
        """
        Extract location (city) from the chatbot's response.
        Example: If the response contains a city name like 'New York', return 'New York'.
        """
        # Ensure response is a string before processing
        if not isinstance(response, str):
            print(f"Error: Response is not a string! {response}")  # Debugging line
            return None

        print(f"Response type: {type(response)}")  # Debugging line

        # Check if locations are in the context
        if 'locations' not in self.context['entities']:
            print("No locations found in context")
            return None

        locations = self.context['entities']['locations']

        # Handle if locations is a list
        if isinstance(locations, list):
            for location in locations:
                # Ensure each location is a string before calling .lower()
                if isinstance(location, str):
                    print(f"Location type: {type(location)}")  # Debugging line
                    if location.lower() in response.lower():
                        return location
                else:
                    print(f"Error: Found a non-string location! {location}")  # Debugging line
        # Handle if locations is a string (should be a list, but just in case)
        elif isinstance(locations, str):
            if locations.lower() in response.lower():
                return locations
        else:
            print(f"Error: locations is neither a list nor a string: {type(locations)}")

        return None

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


if __name__ == "__main__":
    print("Starting the Enhanced ChatBot...")

    # Initialize the EnhancedChatBot
    try:
        chatbot = EnhancedChatBot()
        print("ChatBot initialized successfully!")

        # Example conversation
        user_input = "Tell me about New York"
        response = chatbot.get_response(user_input)
        print(f"User: {user_input}")
        print(f"Bot: {response}")

    except Exception as e:
        print(f"Error during initialization or conversation: {e}")