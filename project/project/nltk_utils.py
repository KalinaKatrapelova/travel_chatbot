import nltk
import numpy as np

# Download both punkt and punkt_tab
# try:
#     nltk.download('punkt')
#     nltk.download('punkt_tab')
# except:
#     # If downloading through code doesn't work, we can try an alternative approach
#     nltk.download('all')  # This will download all NLTK data, including punkt and punkt_tab

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# sentence = ['hello', 'how', 'are', 'you']
# words = ['hi', 'hello', 'I', 'you', 'bye', 'thank', 'cool']
# bag = bag_of_words(sentence, words)
# print(bag)


# Test the functionality
# a = "How long does shipping take?"
# print("Original sentence:", a)
# a = tokenize(a)
# print("Tokenized result:", a)
#
# words = ['Organize', "organizes", 'organizing']
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)