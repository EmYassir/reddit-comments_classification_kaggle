import nltk
import re
import numpy as np
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


class TextProcessingUtil:

    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words('english'))
        self._tokenizer = RegexpTokenizer(r'\w+')
        self._vocabulary = {}
        self._sorted_vocabulary = None
        self._frequent_words_amount = 2000
        self._freq_words = None
        self.labels = None

    def _clean_urls(self, sentence):
        return re.sub(r"http\S+", "", sentence)

    def _remove_stop_words(self, tokenized_sentence):
        return [w for w in tokenized_sentence if w not in self._stop_words]

    def _apply_lemmatizer(self, tokenized_sentence):
        lemmas = []
        for token in tokenized_sentence:
            lemma = self._lemmatizer.lemmatize(token, pos="n")
            lemmas.append(self._lemmatizer.lemmatize(token, pos="v") if lemma == token else lemma)
            if lemma not in self._vocabulary.keys():
                self._vocabulary[lemma] = 1
            else:
                self._vocabulary[lemma]  += 1
        return lemmas

    def _preprocess_sentence(self, sentence):
        step1 = sentence.lower()
        step2 = self._clean_urls(step1)
        step3 = self._tokenizer.tokenize(step2)
        step4 = self._remove_stop_words(step3)
        step5 = self._apply_lemmatizer(step4)
        return step5

    def _get_bow_representation(self, tokenized_sentence):
        if not self._sorted_vocabulary:
            self._sorted_vocabulary = sorted(self._freq_words)
        bow = np.zeros(len(self._sorted_vocabulary))
        for token in tokenized_sentence:
            try:
                index = self._sorted_vocabulary.index(token)
                bow[index] += 1
            except ValueError:
                pass
        return bow

    def _get_numeric_labels(self, label_values):
        self.labels = np.unique(label_values)
        return list(map((lambda x: np.where(self.labels == x)[0][0]), label_values))

    def get_vocabulary(self):
        return self._sorted_vocabulary

    def get_bow_matrix(self, sentences, label_values=None):
        tokenized_sentences = list(map(self._preprocess_sentence, sentences))
        self._freq_words = heapq.nlargest(self._frequent_words_amount, self._vocabulary, key=self._vocabulary.get)
        bow_reps = list(map(self._get_bow_representation, tokenized_sentences))
        return np.column_stack((bow_reps, self._get_numeric_labels(label_values))) if label_values else np.array(bow_reps)
