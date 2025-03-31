import torch
import json
import random
import os
from model import EnhancedNeuralNet
from nltk_utils import tokenize, get_sentence_embedding, correct_spelling, extract_entities, load_embeddings
from intent_handler import IntentHandler
from dynamic_content import DynamicContentProvider


class AdvancedTravelBot:
    """
    Advanced travel chatbot that integrates improved intent handling,
    dynamic content, and contextual conversation management.
    """

    def __init__(self, model_weights='enhanced_model_weights.pth',
                 model_data='enhanced_model_data.json',
                 intents_file='intents.json',
                 confidence_threshold=0.6):
        """
        Initialize the advanced travel chatbot.

        Args:
            model_weights: Path to the trained model weights
            model_data: Path to the model data JSON file
            intents_file: Path to the intents JSON file
            confidence_threshold: Confidence threshold for intent matching
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if files exist
        if not os.path.exists(intents_file):
            raise FileNotFoundError(f"Intents file '{intents_file}' not found.")
        if not os.path.exists(model_data):
            raise FileNotFoundError(f"Model data file '{model_data}' not found.")
        if not os.path.exists(model_weights):
            raise FileNotFoundError(f"Model weights file '{model_weights}' not found.")

        # Load intents
        try:
            with open(intents_file, 'r') as f:
                self.intents = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in '{intents_file}'.")
        except Exception as e:
            raise Exception(f"Error loading intents file: {str(e)}")

        # Load model data
        try:
            with open(model_data, 'r') as f:
                model_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in '{model_data}'.")
        except Exception as e:
            raise Exception(f"Error loading model data: {str(e)}")

        # Get model parameters
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.embedding_dim = model_data.get('embedding_dim', 100)
        self.tags = model_data['tags']

        # Load word embeddings
        print("Loading word embeddings...")
        load_embeddings()

        # Initialize the model
        try:
            self.model = EnhancedNeuralNet(
                input_size=None,  # Not needed for embedding model
                hidden_size=self.hidden_size,
                num_classes=self.output_size,
                embedding_dim=self.embedding_dim
            ).to(self.device)

            # Load model weights
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device, weights_only=True))
            self.model.eval()
            print(f"Model loaded from {model_weights}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

        # Initialize components
        self.intent_handler = IntentHandler(
            model=self.model,
            tags=self.tags,
            confidence_threshold=confidence_threshold
        )

        self.dynamic_content = DynamicContentProvider()

        # Initialize conversation context
        self.reset_context()

        print(f"Advanced TravelBot initialized on {self.device}")

    def reset_context(self):
        """Reset the conversation context."""
        self.context = {
            'current_city': None,
            'current_topic': None,
            'entities': {
                'locations': [],
                'dates': [],
                'landmarks': [],
                'travel_info': []
            },
            'last_intent': None,
            'disambiguation_active': False,
            'disambiguation_options': [],
            'conversation_history': [],
            'suggested_followups': []
        }

    def preprocess_input(self, user_input):
        """
        Preprocess the user input by correcting spelling and extracting entities.

        Args:
            user_input: Raw user input string

        Returns:
            tuple: Processed text and entities
        """
        # Correct spelling
        corrected_text = correct_spelling(user_input)

        # Extract entities
        entities = extract_entities(corrected_text)

        # Update conversation context with entities
        if entities['locations']:
            self.context['current_city'] = entities['locations'][0]
            self.context['entities']['locations'].extend(
                [loc for loc in entities['locations'] if loc not in self.context['entities']['locations']]
            )

        self.context['entities']['landmarks'].extend(
            [landmark for landmark in entities['landmarks']
             if landmark not in self.context['entities']['landmarks']]
        )

        self.context['entities']['dates'].extend(
            [date for date in entities['dates']
             if date not in self.context['entities']['dates']]
        )

        return corrected_text, entities

    def handle_disambiguation_response(self, user_input):
        """
        Handle user response to a disambiguation question.

        Args:
            user_input: User's response to disambiguation

        Returns:
            str: Response based on disambiguation resolution
        """
        # Simple keyword matching to resolve disambiguation
        user_input_lower = user_input.lower()

        # Check for city mentions
        cities = ["paris", "london", "tokyo", "rome", "barcelona", "new york"]
        mentioned_city = None
        for city in cities:
            if city in user_input_lower:
                mentioned_city = city
                break

        # Check for topic mentions
        topics = {
            "landmark": ["landmark", "attraction", "sight", "see", "visit", "place"],
            "food": ["food", "eat", "restaurant", "cuisine", "dish"],
            "weather": ["weather", "climate", "temperature", "season"],
            "transportation": ["transport", "travel", "get around"]
        }

        mentioned_topic = None
        for topic, keywords in topics.items():
            if any(keyword in user_input_lower for keyword in keywords):
                mentioned_topic = topic
                break

        # Update context with disambiguation results
        if mentioned_city:
            self.context['current_city'] = mentioned_city

        if mentioned_topic:
            self.context['current_topic'] = mentioned_topic

        # If both city and topic are identified, construct the intent
        if mentioned_city and mentioned_topic:
            resolved_intent = f"{mentioned_city}_{mentioned_topic}s"
            if resolved_intent in self.tags:
                self.context['last_intent'] = resolved_intent
                self.context['disambiguation_active'] = False
                return self.get_intent_response(resolved_intent)

        # If only city is identified, ask about topic
        elif mentioned_city:
            self.context['disambiguation_active'] = True
            self.context['disambiguation_options'] = ["landmarks", "food", "weather", "transportation"]
            return f"Great! What would you like to know about {mentioned_city.title()}? I can tell you about attractions, local food, weather, or how to get around."

        # If only topic is identified, ask about city
        elif mentioned_topic:
            self.context['disambiguation_active'] = True
            self.context['disambiguation_options'] = cities
            cities_list = ", ".join([city.title() for city in cities[:-1]]) + f", or {cities[-1].title()}"
            return f"I can provide information about {mentioned_topic}s in several cities. Which city are you interested in? We cover {cities_list}."

        # If neither is identified, provide a generic response
        self.context['disambiguation_active'] = False
        return "I'm still not sure what you're looking for. Let's start over. What city are you planning to visit?"

    def get_dynamic_content(self, intent, entities):
        """
        Get dynamic content based on intent and entities.

        Args:
            intent: The identified intent
            entities: Extracted entities from user input

        Returns:
            str: Dynamic content to enhance the response
        """
        # If no city is identified, we can't provide dynamic content
        if not self.context['current_city'] and not entities['locations']:
            return ""

        # Use context city or the first location in entities
        city = self.context['current_city'] or entities['locations'][0]

        # Weather information
        if 'weather' in intent:
            try:
                weather_data = self.dynamic_content.get_weather(city)
                if weather_data.get('source') == 'mock data':
                    return f"\n\nCurrent weather in {city.title()}: {weather_data['temperature']['current']}°C, {weather_data['conditions']}."
                else:
                    return f"\n\nCurrent weather update for {city.title()}: {weather_data['temperature']['current']}°C, {weather_data['conditions']}. Humidity: {weather_data['humidity']}%. Wind: {weather_data['wind_speed']} m/s."
            except Exception:
                return ""

        # Events information
        elif 'landmarks' in intent or 'attractions' in intent:
            try:
                events = self.dynamic_content.get_events(city, limit=3)
                if events:
                    events_text = "\n\nUpcoming events you might be interested in:"
                    for event in events:
                        events_text += f"\n- {event['name']} ({event['date']}): {event['price']}"
                    return events_text
            except Exception:
                return ""

        # Exchange rate information for budget queries
        elif 'cost' in intent or 'budget' in intent:
            try:
                exchange = self.dynamic_content.get_exchange_rate("USD", "EUR")
                return f"\n\nCurrent exchange rate: 1 USD = {exchange['rate']} EUR (as of {exchange['timestamp']})."
            except Exception:
                return ""

        return ""

    def generate_suggested_followups(self, intent):
        """
        Generate suggested follow-up questions based on the current intent.

        Args:
            intent: The current intent

        Returns:
            list: Suggested follow-up questions
        """
        followups = []

        # Extract city from intent if present
        city = None
        for supported_city in ["paris", "london", "tokyo", "rome", "barcelona", "new york"]:
            if supported_city in intent:
                city = supported_city
                break

        # If no city was found, use the context city
        if not city:
            city = self.context['current_city']

        # If we have a city, generate city-specific followups
        if city:
            city_title = city.title()

            # Different followups based on intent type
            if 'landmarks' in intent:
                followups = [
                    f"What food should I try in {city_title}?",
                    f"How's the weather in {city_title}?",
                    f"How do I get around in {city_title}?",
                    f"What are some hidden gems in {city_title}?"
                ]
            elif 'food' in intent:
                followups = [
                    f"What are the must-see attractions in {city_title}?",
                    f"What areas have the best restaurants in {city_title}?",
                    f"Any local street food recommendations for {city_title}?",
                    f"What's the typical budget for dining in {city_title}?"
                ]
            elif 'weather' in intent:
                followups = [
                    f"When is the best time to visit {city_title}?",
                    f"What should I pack for {city_title} in summer/winter?",
                    f"What indoor activities are there in {city_title} if it rains?",
                    f"What landmarks should I visit in {city_title}?"
                ]
            else:
                followups = [
                    f"What are the top attractions in {city_title}?",
                    f"What food should I try in {city_title}?",
                    f"What's the weather like in {city_title}?",
                    f"How do I get around in {city_title}?"
                ]
        else:
            # General followups if no city is identified
            followups = [
                "What city are you planning to visit?",
                "Would you like information about Paris, London, or Tokyo?",
                "Are you looking for information about attractions, food, or weather?",
                "Do you need budget travel tips?"
            ]

        # Randomize order and limit to 2 suggestions
        random.shuffle(followups)
        return followups[:2]

    def get_intent_response(self, intent):
        """
        Get a response for a given intent from the intents file.

        Args:
            intent: The intent tag

        Returns:
            str: Response for the intent
        """
        # Look for the intent in intents file
        for intent_data in self.intents['intents']:
            if intent_data['tag'] == intent:
                return random.choice(intent_data['responses'])

        # If intent not found, use the fallback intent
        for intent_data in self.intents['intents']:
            if intent_data['tag'] == 'fallback':
                return random.choice(intent_data['responses'])

        # If no fallback intent exists, return a generic response
        return "I'm sorry, I don't have information about that yet. I can help with popular cities like Paris, London, and Tokyo."

    def format_response_with_followups(self, response, suggested_followups):
        """
        Format the response with suggested followups.

        Args:
            response: Main response text
            suggested_followups: List of suggested followup questions

        Returns:
            str: Formatted response with followups
        """
        if not suggested_followups:
            return response

        formatted = response + "\n\nYou might also want to ask:"
        for i, followup in enumerate(suggested_followups, 1):
            formatted += f"\n{i}. {followup}"

        return formatted

    def process(self, user_input):
        """
        Process user input and generate a response.

        Args:
            user_input: Raw user input string

        Returns:
            str: Chatbot response
        """
        # Handle active disambiguation
        if self.context['disambiguation_active']:
            response = self.handle_disambiguation_response(user_input)

            # Generate suggested followups if disambiguation is resolved
            if not self.context['disambiguation_active']:
                suggested_followups = self.generate_suggested_followups(self.context['last_intent'] or '')
                self.context['suggested_followups'] = suggested_followups
                response = self.format_response_with_followups(response, suggested_followups)

            # Add to conversation history
            self.context['conversation_history'].append({
                'user_input': user_input,
                'response': response
            })

            return response

        # Regular processing
        processed_text, entities = self.preprocess_input(user_input)

        # Get sentence embedding
        embedding = get_sentence_embedding(processed_text)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Use intent handler for classification and possible disambiguation
        intent_result = self.intent_handler.handle_intent(embedding_tensor, processed_text, self.context)

        # Update context
        self.context['last_intent'] = intent_result['tag']

        # Handle disambiguation if needed
        if intent_result['needs_disambiguation']:
            self.context['disambiguation_active'] = True
            self.context['disambiguation_options'] = [option[0] for option in intent_result['disambiguation_options']]
            response = self.intent_handler.generate_disambiguation_response(intent_result)

            # Add to conversation history
            self.context['conversation_history'].append({
                'user_input': user_input,
                'response': response
            })

            return response

        # Handle fallback if needed
        if intent_result['is_fallback']:
            response = self.intent_handler.generate_fallback_response(processed_text)

            # Add to conversation history
            self.context['conversation_history'].append({
                'user_input': user_input,
                'response': response
            })

            return response

        # Get response for the identified intent
        base_response = self.get_intent_response(intent_result['tag'])

        # Add dynamic content if applicable
        dynamic_content = self.get_dynamic_content(intent_result['tag'], entities)
        response = base_response + dynamic_content

        # Generate suggested followups
        suggested_followups = self.generate_suggested_followups(intent_result['tag'])
        self.context['suggested_followups'] = suggested_followups

        # Format response with followups
        final_response = self.format_response_with_followups(response, suggested_followups)

        # Add to conversation history
        self.context['conversation_history'].append({
            'user_input': user_input,
            'response': final_response
        })

        return final_response


def chat():
    """Run an interactive chat session with the advanced travel chatbot."""
    print("Starting the Advanced TravelBot...")

    try:
        chatbot = AdvancedTravelBot()

        print("\n" + "=" * 60)
        print("Welcome to the Advanced TravelBot!")
        print("I can help you with detailed information about Paris, London, Tokyo, Rome, and more.")
        print("Ask me about landmarks, local food, weather, hidden gems, and travel tips.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("=" * 60 + "\n")

        # Main conversation loop
        while True:
            # Get user input
            user_input = input("You: ")

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nTravelBot: Goodbye! Safe travels and have a wonderful journey!")
                break

            # Process input and get response
            try:
                response = chatbot.process(user_input)
                print(f"\nTravelBot: {response}\n")
            except Exception as e:
                print(f"\nTravelBot: I'm having trouble processing that. Could you try asking in a different way?\n")
                print(f"Error: {str(e)}")

    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return


if __name__ == "__main__":
    chat()