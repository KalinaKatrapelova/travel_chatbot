import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import numpy as np
import gensim.downloader as api
import re
import os

# Download necessary NLTK resources with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('words', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download error: {e}")
    print("Some NLP features might not work correctly")

# Initialize stemmer and spell checker
stemmer = PorterStemmer()
spell = SpellChecker()

# Global variable to store word vectors
word_vectors = None

# Add travel-specific terms to the spell checker dictionary
travel_terms = ['Paris', 'London', 'Tokyo', 'Berlin', 'Rome', 'Barcelona', 'Madrid',
                'Eiffel', 'Tower', 'Louvre', 'Colosseum', 'Statue', 'Liberty']

# Custom mapping for common misspellings in travel domain
custom_corrections = {
    'pariz': 'paris',
    'parris': 'paris',
    'londun': 'london',
    'tokio': 'tokyo',
    'eifel': 'eiffel',
    'statue of libety': 'statue of liberty'
}


def load_embeddings(embedding_type='glove-wiki-gigaword-100'):
    """Load word embeddings model"""
    global word_vectors
    try:
        word_vectors = api.load(embedding_type)
        print(f"Loaded {embedding_type} embeddings")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("Using random embeddings instead")
        # Create a simple random embedding dictionary for testing
        from gensim.models import Word2Vec

        # Generate some random words and vectors
        vocab = set()
        try:
            with open('intents.json', 'r') as f:
                import json
                intents = json.load(f)
                for intent in intents['intents']:
                    for pattern in intent['patterns']:
                        words = tokenize(pattern)
                        vocab.update(words)
        except Exception as e:
            print(f"Warning: Could not load intents.json: {e}")
            # If intents.json doesn't exist, use some sample words
            vocab = {"travel", "visit", "city", "when", "where", "how", "paris", "london", "tokyo"}

        # Create a simple embedding model
        sentences = [[word] for word in vocab]
        word_vectors = Word2Vec(sentences, vector_size=100, min_count=1, window=5).wv
        print("Using simple random embeddings")


def tokenize(sentence):
    """Tokenize a sentence into words"""
    return word_tokenize(sentence.lower())


def stem(word):
    """Stem a word to its root form"""
    return stemmer.stem(word.lower())


def correct_spelling(sentence):
    """Correct spelling mistakes in the input sentence"""
    # Add travel terms to the spell checker's dictionary
    for term in travel_terms:
        spell.word_frequency.load_words([term])

    # First, check for custom travel-specific corrections
    lower_sentence = sentence.lower()
    for misspelled, correct in custom_corrections.items():
        lower_sentence = lower_sentence.replace(misspelled, correct)

    words = tokenize(lower_sentence)
    corrected_words = []

    for word in words:
        # Only try to correct words that might be misspelled
        if word.isalpha() and (word_vectors is None or word not in word_vectors) and word not in stopwords.words('english'):
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
        else:
            corrected_words.append(word)

    return ' '.join(corrected_words)


def extract_entities(sentence):
    """Extract named entities from the sentence using pattern matching"""
    entities = {
        'locations': [],
        'dates': [],
        'landmarks': [],
        'travel_info': []
    }

    # Cities to detect
    cities = ['Paris', 'London', 'Tokyo', 'New York', 'Rome', 'Berlin', 'Barcelona', 'Madrid',
              'Amsterdam', 'Vienna', 'Prague', 'Budapest', 'Athens', 'Dubai', 'Istanbul']

    # Landmarks to detect
    landmarks = {
        'Eiffel Tower': 'Paris',
        'Louvre': 'Paris',
        'Big Ben': 'London',
        'Tower Bridge': 'London',
        'Tokyo Tower': 'Tokyo',
        'Shibuya Crossing': 'Tokyo',
        'Statue of Liberty': 'New York',
        'Empire State': 'New York',
        'Colosseum': 'Rome',
        'Vatican': 'Rome',
        'Brandenburg Gate': 'Berlin',
        'Sagrada Familia': 'Barcelona'
    }

    # Check for cities
    for city in cities:
        if re.search(r'\b' + city + r'\b', sentence, re.IGNORECASE):
            if city.lower() not in [loc.lower() for loc in entities['locations']]:
                entities['locations'].append(city)

    # Check for landmarks
    for landmark, city in landmarks.items():
        if re.search(r'\b' + landmark.replace(' ', r'\s+') + r'\b', sentence, re.IGNORECASE):
            entities['landmarks'].append(landmark)
            # Add the associated city if not already there
            if city not in entities['locations']:
                entities['locations'].append(city)

    # Pattern matching for dates
    date_patterns = [
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        r'\b(?:next|this|coming|last)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
    ]

    for pattern in date_patterns:
        dates = re.findall(pattern, sentence, re.IGNORECASE)
        entities['dates'].extend(dates)

    # Detect travel-related information like transportation, accommodation, etc.
    travel_keywords = {
        'transportation': ['flight', 'train', 'bus', 'taxi', 'subway', 'metro', 'rent a car', 'airport'],
        'accommodation': ['hotel', 'hostel', 'airbnb', 'stay', 'room', 'book', 'reservation'],
        'activities': ['visit', 'see', 'tour', 'museum', 'beach', 'mountain', 'hike', 'restaurant']
    }

    for category, keywords in travel_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', sentence, re.IGNORECASE):
                entities['travel_info'].append((keyword, category))

    return entities


def get_word_embedding(word):
    """Get the embedding vector for a word"""
    if word_vectors is None:
        load_embeddings()

    # Try to get the word vector
    try:
        return word_vectors[word.lower()]
    except KeyError:
        # If word is not in vocabulary, try stemmed version
        stemmed = stem(word)
        try:
            return word_vectors[stemmed]
        except KeyError:
            # If still not found, return zero vector
            return np.zeros(word_vectors.vector_size)


def get_sentence_embedding(sentence):
    """
    Convert a sentence to a single embedding vector by averaging word vectors
    """
    if word_vectors is None:
        load_embeddings()

    words = tokenize(sentence)

    # Filter out stopwords for better representation
    words = [word for word in words if word not in stopwords.words('english')]

    if not words:
        # Return zero vector if no words left after filtering
        return np.zeros(word_vectors.vector_size)

    # Get embeddings for each word and average them
    word_embeddings = [get_word_embedding(word) for word in words]
    return np.mean(word_embeddings, axis=0)


def bag_of_words(tokenized_sentence, all_words):
    """
    Legacy bag of words method - kept for backward compatibility
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


# Test the functionality
if __name__ == "__main__":
    # Load embeddings during import
    print("Loading word embeddings...")
    load_embeddings()

    # Test sentence
    test = "I want to visit Pariz next month and see the Eiffel Tower"

    print("Original sentence:", test)

    # Test spell correction
    corrected = correct_spelling(test)
    print("Spell-corrected:", corrected)

    # Test entity extraction
    entities = extract_entities(corrected)
    print("Extracted entities:", entities)

    # Test embeddings
    embedding = get_sentence_embedding(corrected)
    print("Sentence embedding shape:", embedding.shape)
    print("Embedding sample:", embedding[:5])  # Show first 5 values