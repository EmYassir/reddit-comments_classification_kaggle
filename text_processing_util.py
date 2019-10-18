import nltk
import re
import numpy as np
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


class TextProcessingUtil:

    def __init__(self, frequent_words_amount=2500, frequent_words_per_label=1010, frequent_words_to_skip=0):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self._lemmatizer = WordNetLemmatizer()
        self._stemmer = SnowballStemmer("english")
        self._stop_words = set(stopwords.words('english')),
        self._tokenizer = RegexpTokenizer(r'\w+')
        self._frequent_words_amount = frequent_words_amount
        self._frequent_words_per_label = frequent_words_per_label
        self._frequent_words_to_skip = frequent_words_to_skip
        self._vocabulary = {}
        self._general_vocabulary = {}
        self._unique_vocabulary = {}
        self._unique_words = {}
        self._sorted_vocabulary = None
        self._freq_words = []
        self._labels = None
        self._numeric_train_labels = None

    """
    Text pre-processing utilitary methods
    """
    def _clean_urls(self, sentence):
        return re.sub(r"http\S+", "", sentence)

    def _remove_stop_words(self, tokenized_sentence):
        return [w for w in tokenized_sentence if w not in self._stop_words]

    def _apply_stemmer(self, tokenized_sentence):
        stems = []
        for token in tokenized_sentence:
            stem = self._stemmer.stem(token)
            stems.append(stem)
        return stems

    def _apply_noun_lemmatizer(self, tokenized_sentence):
        lemmas = []
        for token in tokenized_sentence:
            lemma = self._lemmatizer.lemmatize(token, pos="n")
            lemmas.append(lemma)
        return lemmas

    def _extract_numeric_words(self, tokenized_sentence):
        alphas = []
        for token in tokenized_sentence:
            try:
                float(token)
            except ValueError:
                alphas.append(token)
        return alphas

    def _add_to_vocabulary(self, current_label, tokenized_sentence):
        if current_label not in self._vocabulary.keys():
            self._vocabulary[current_label] = {}

        for token in tokenized_sentence:
            if token not in self._vocabulary[current_label].keys():
                self._vocabulary[current_label][token] = 1
                self._general_vocabulary[token] = 1
            else:
                self._vocabulary[current_label][token] += 1
                self._general_vocabulary[token] = 1
        return np.array(tokenized_sentence)

    def _preprocess_sentence(self, current_label, sentence):
        res = sentence.lower()
        res = self._clean_urls(res)
        res = self._tokenizer.tokenize(res)
        res = self._remove_stop_words(res)
        res = self._extract_numeric_words(res)
        res = self._apply_noun_lemmatizer(res)
        res = self._apply_stemmer(res)
        return np.array(res) if current_label is None else self._add_to_vocabulary(current_label, res)

    """
    Bag of words representation building methods
    """
    def _get_bow_representation(self, tokenized_sentence):
        if not self._sorted_vocabulary:
            self._sorted_vocabulary = sorted(set(self._freq_words))
        bow = np.zeros(len(self._sorted_vocabulary))
        for token in tokenized_sentence:
            try:
                index = self._sorted_vocabulary.index(token)
                bow[index] += 1
            except ValueError:
                pass
        return bow

    def _set_frequent_unique_words(self):
        commons_set = None
        for c in range(self._labels.shape[0]):
            self._unique_words[c] = np.array(list(self._vocabulary[c].keys()))
            commons_set = self._unique_words[c] if commons_set is None else np.intersect1d(commons_set, self._unique_words[c], assume_unique=True)

        for c in self._unique_words.keys():
            self._unique_words[c] = np.setdiff1d(self._unique_words[c], commons_set, assume_unique=True)
            self._unique_vocabulary[c] = {}
            for w in self._unique_words[c]:
                self._unique_vocabulary[c][w] = self._vocabulary[c][w]
            self._freq_words += heapq.nlargest(self._frequent_words_per_label, self._unique_vocabulary[c], key=self._unique_vocabulary[c].get)

        self._freq_words += heapq.nlargest(self._frequent_words_amount, self._general_vocabulary, key=self._general_vocabulary.get)

    """
    Utilitary getters and setters
    """
    def get_vocabulary(self):
        return np.array(self._sorted_vocabulary)

    def get_labels(self):
        return self._labels

    def _get_numeric_label(self, index):
        return None if self._numeric_train_labels is None else self._numeric_train_labels[index]

    def _set_numeric_labels(self, label_values):
        if label_values is not None:
            self._labels = np.unique(label_values)
            self._numeric_train_labels = np.array(list(map((lambda x: np.where(self._labels == x)[0][0]), label_values)))

    def get_bow_matrix(self, sentences, label_values=None):
        self._set_numeric_labels(label_values)
        tokenized_sentences = np.array([self._preprocess_sentence(self._get_numeric_label(i), sentence) for i, sentence in enumerate(sentences)])
        self._set_frequent_unique_words()
        bow_reps = list(map(self._get_bow_representation, tokenized_sentences))
        return np.column_stack((bow_reps, self._numeric_train_labels)) if label_values else np.array(bow_reps)
