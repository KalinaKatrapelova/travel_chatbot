import torch
import numpy as np
from nltk.tokenize import word_tokenize
import re


class IntentHandler:
    """
    Handles intent disambiguation and provides fallback mechanisms
    when the model is uncertain about the intent.
    """

    def __init__(self, model, tags, confidence_threshold=0.6, disambiguation_threshold=0.2):
        """
        Initialize the intent handler.

        Args:
            model: The trained neural network model
            tags: List of intent tags the model was trained on
            confidence_threshold: Minimum confidence required for a definitive match
            disambiguation_threshold: Threshold for considering multiple intents
        """
        self.model = model
        self.tags = tags
        self.confidence_threshold = confidence_threshold
        self.disambiguation_threshold = disambiguation_threshold

        # Group related intents for better disambiguation
        self.intent_groups = {
            "city_landmarks": ["paris_landmarks", "london_landmarks", "tokyo_landmarks",
                               "rome_landmarks", "barcelona_landmarks"],
            "city_food": ["paris_food", "london_food", "tokyo_food", "rome_food", "barcelona_food"],
            "city_weather": ["paris_weather", "london_weather", "tokyo_weather"],
            "general": ["greeting", "goodbye", "help", "continue_conversation"]
        }

        # Create reverse mapping from intent to group
        self.intent_to_group = {}
        for group, intents in self.intent_groups.items():
            for intent in intents:
                self.intent_to_group[intent] = group

        # Cities supported by the chatbot
        self.supported_cities = ["paris", "london", "tokyo", "rome", "barcelona", "new york"]

        # Keywords associated with intent categories
        self.intent_keywords = {
            "landmarks": ["see", "visit", "attraction", "landmark", "sight", "monument", "museum", "place"],
            "food": ["eat", "food", "restaurant", "cuisine", "dish", "meal", "dining", "taste"],
            "weather": ["weather", "climate", "temperature", "rain", "sunny", "cold", "hot", "season"],
            "transportation": ["transport", "get around", "metro", "bus", "train", "taxi", "walking", "travel"]
        }

    def extract_city(self, query):
        """
        Extract city name from user query if present.
        """
        query_lower = query.lower()
        for city in self.supported_cities:
            if city in query_lower:
                return city
        return None

    def extract_intent_type(self, query):
        """
        Extract intent type (landmarks, food, etc.) from query based on keywords.
        """
        query_lower = query.lower()

        for intent_type, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent_type
        return None

    def get_top_intents(self, embedding_tensor, top_k=3):
        """
        Get the top-k intents with their probabilities.

        Args:
            embedding_tensor: Input embedding tensor
            top_k: Number of top intents to return

        Returns:
            list: Top intents with their probabilities
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(embedding_tensor)

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)[
                0] if outputs.dim() > 1 else torch.nn.functional.softmax(outputs, dim=0)

            # Get top-k indices and their probabilities
            if probs.dim() == 0:  # Handle single-value case
                top_probs, top_indices = torch.tensor([probs.item()]), torch.tensor([0])
            else:
                top_probs, top_indices = torch.topk(probs, min(top_k, len(self.tags)))

            # Convert to list of (tag, probability) tuples
            top_intents = [(self.tags[top_indices[i].item()], top_probs[i].item()) for i in range(len(top_indices))]

        return top_intents

    def handle_intent(self, embedding_tensor, user_query, context=None):
        """
        Handle intent classification with disambiguation and fallbacks.

        Args:
            embedding_tensor: Input embedding tensor
            user_query: Original user query string
            context: Conversation context (optional)

        Returns:
            dict: Intent handling result with tag, response, and any disambiguation info
        """
        # Get top intents
        top_intents = self.get_top_intents(embedding_tensor)

        # Extract city and intent type from query
        extracted_city = self.extract_city(user_query)
        extracted_intent_type = self.extract_intent_type(user_query)

        # Check if we have a high-confidence match
        top_tag, top_prob = top_intents[0]

        result = {
            "tag": top_tag,
            "confidence": top_prob,
            "needs_disambiguation": False,
            "is_fallback": False,
            "disambiguation_options": [],
            "extracted_info": {
                "city": extracted_city,
                "intent_type": extracted_intent_type
            }
        }

        # Case 1: High confidence match
        if top_prob >= self.confidence_threshold:
            return result

        # Case 2: Multiple possible intents with similar probabilities
        second_prob = top_intents[1][1] if len(top_intents) > 1 else 0
        if (top_prob - second_prob) < self.disambiguation_threshold:
            # Check if intents are from different groups
            if len(top_intents) > 1:
                top_group = self.intent_to_group.get(top_tag, "other")
                second_tag = top_intents[1][0]
                second_group = self.intent_to_group.get(second_tag, "other")

                if top_group != second_group:
                    # Need disambiguation between different intent types
                    result["needs_disambiguation"] = True
                    result["disambiguation_options"] = top_intents[:3]  # Top 3 options for disambiguation

        # Case 3: Low confidence across all intents - use fallback
        if top_prob < 0.4:  # Very low confidence threshold
            result["is_fallback"] = True

            # Try to construct a better fallback using extracted information
            if extracted_city and extracted_intent_type:
                # If we have both city and intent type, try to find a matching intent
                potential_intent = f"{extracted_city}_{extracted_intent_type}"
                if potential_intent in self.tags:
                    result["tag"] = potential_intent
                    result["is_fallback"] = False

            # If we only have city, suggest possible intents for that city
            elif extracted_city:
                result["extracted_info"]["city"] = extracted_city
                result["needs_disambiguation"] = True

                # Filter intents related to the extracted city
                city_intents = [tag for tag in self.tags if extracted_city in tag]
                result["disambiguation_options"] = [(tag, 0.0) for tag in city_intents]

            # If we only have intent type, suggest possible cities for that intent
            elif extracted_intent_type:
                result["extracted_info"]["intent_type"] = extracted_intent_type
                result["needs_disambiguation"] = True

                # Suggest all cities for this intent type
                result["disambiguation_options"] = [
                    (f"{city}_{extracted_intent_type}", 0.0) for city in self.supported_cities
                    if f"{city}_{extracted_intent_type}" in self.tags
                ]

        return result

    def generate_disambiguation_response(self, disambiguation_result):
        """
        Generate a response to help disambiguate user intent.

        Args:
            disambiguation_result: Result from handle_intent containing disambiguation info

        Returns:
            str: Disambiguation response
        """
        options = disambiguation_result["disambiguation_options"]
        extracted_city = disambiguation_result["extracted_info"]["city"]
        extracted_intent_type = disambiguation_result["extracted_info"]["intent_type"]

        if extracted_city and not extracted_intent_type:
            return f"I see you're interested in {extracted_city.title()}. Would you like to know about attractions, food, or weather there?"

        elif extracted_intent_type and not extracted_city:
            cities_list = ", ".join(
                [city.title() for city in self.supported_cities[:-1]]) + f", or {self.supported_cities[-1].title()}"
            return f"I can provide information about {extracted_intent_type} in various cities. Which city are you interested in? We cover {cities_list}."

        elif options:
            # Create a readable list of options
            intent_descriptions = {
                "landmarks": "attractions and sights",
                "food": "local cuisine and restaurants",
                "weather": "weather and best time to visit",
                "transportation": "how to get around"
            }

            option_texts = []
            for option, _ in options:
                # Parse the option to create a readable description
                parts = option.split('_')
                if len(parts) >= 2:
                    city = parts[0].title()
                    intent_type = '_'.join(parts[1:])
                    description = intent_descriptions.get(intent_type, intent_type)
                    option_texts.append(f"{city} {description}")
                else:
                    option_texts.append(option)

            options_text = ", ".join(option_texts[:-1])
            if len(option_texts) > 1:
                options_text += f", or {option_texts[-1]}"
            else:
                options_text = option_texts[0]

            return f"I'm not quite sure what you're asking. Are you interested in {options_text}?"

        return "I'm not sure I understood your question. Could you provide more details about what city or information you're interested in?"

    def generate_fallback_response(self, query=None):
        """
        Generate a fallback response when the model can't determine the intent.

        Args:
            query: The original query for contextual fallbacks

        Returns:
            str: Fallback response
        """
        generic_fallbacks = [
            "I'm not sure I understood that. I can help with information about landmarks, food, weather, and transportation in cities like Paris, London, and Tokyo. What would you like to know?",
            "I don't have enough information to answer that properly. Could you tell me which city you're interested in?",
            "I'm here to help with travel information for major cities. Could you rephrase your question to specify what you're looking for?",
            "I might have missed something. I know about attractions, local cuisine, and travel tips for several cities. How can I help you plan your trip?"
        ]

        # Check for city mentions even in unclear queries
        extracted_city = self.extract_city(query) if query else None

        if extracted_city:
            return f"I see you mentioned {extracted_city.title()}. I can tell you about attractions, food, or weather there. What would you like to know?"

        return np.random.choice(generic_fallbacks)


# Example of how to use this class in your chatbot
if __name__ == "__main__":
    # Mock model and tags for demonstration
    class MockModel:
        def eval(self):
            pass

        def __call__(self, x):
            # Return mock output probabilities
            return torch.tensor([0.2, 0.1, 0.4, 0.3])


    mock_model = MockModel()
    mock_tags = ["greeting", "goodbye", "paris_landmarks", "london_food"]

    # Initialize the intent handler
    handler = IntentHandler(mock_model, mock_tags)

    # Example usage
    mock_embedding = torch.tensor([[0.1, 0.2, 0.3]])  # Mock embedding vector
    result = handler.handle_intent(mock_embedding, "What can I see in Paris?")

    print("Intent handling result:", result)

    if result["needs_disambiguation"]:
        disambiguation_response = handler.generate_disambiguation_response(result)
        print("Disambiguation response:", disambiguation_response)

    if result["is_fallback"]:
        fallback_response = handler.generate_fallback_response("What can I see in Paris?")
        print("Fallback response:", fallback_response)