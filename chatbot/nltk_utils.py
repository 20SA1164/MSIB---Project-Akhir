from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np

# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    # Converting sentence into number of words (tokens)
    # sentence: 'May Force Be With You!'
    # token: ['May', 'Force', 'Be', 'With', 'You', '!']
    return word_tokenize(sentence)


def stem(word):
    # concept -  stemming
    # Generating the root form of the words.
    # words: ["organize", "organizes", "organizing"]
    # stemmed_words: ['organ', 'organ', 'organ']
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["helo", "ada", "yang", "bisa", "saya", "bantu"]
    words = ["hai", "helo", "saya", "Anda", "bye", "terimakasih"]
    bag =   [0,     1,      0,     1,     0,      0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


if __name__ == '__main__':
    # step 1 - tokenization
    # a = "How long does shipping take?"
    # print("sentence: " + a)
    # a = tokenize(a)
    # print(a)

    # step 2 - stemming
    # words = ["organize", "organizes", "organizing"]
    # stemmed_words = [stem(w) for w in words]
    # print(stemmed_words)

    # step 4 - bag of words
    sentence = ["helo", "ada", "yang", "bisa", "saya", "bantu"]
    words = ["hai", "helo", "saya", "Anda", "bye", "terimakasih"]
    bag = bag_of_words(sentence, words)
    print(bag)