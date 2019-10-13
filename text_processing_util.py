import nltk
import re
import numpy as np
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


class TextProcessingUtil:

    def __init__(self, frequent_words_amount=2000, frequent_words_per_label=75):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words('english'))
        self._tokenizer = RegexpTokenizer(r'\w+')
        self._frequent_words_amount = frequent_words_amount
        self._frequent_words_per_label = frequent_words_per_label
        self._vocabulary = {}
        self._sorted_vocabulary = None
        self._freq_words = None
        self._labels = None

    def _clean_urls(self, sentence):
        return re.sub(r"http\S+", "", sentence)

    def _remove_stop_words(self, tokenized_sentence):
        return [w for w in tokenized_sentence if w not in self._stop_words]

    def _apply_verb_lemmatizer(self, tokenized_sentence):
        lemmas = []
        for token in tokenized_sentence:
            lemma = self._lemmatizer.lemmatize(token, pos="v")
            lemmas.append(lemma)
            if lemma not in self._vocabulary.keys():
                self._vocabulary[lemma] = 1
            else:
                self._vocabulary[lemma] += 1
        return lemmas

    def _apply_noun_lemmatizer(self, tokenized_sentence):
        lemmas = []
        for token in tokenized_sentence:
            lemma = self._lemmatizer.lemmatize(token, pos="n")
            lemmas.append(lemma)
        return lemmas

    def _preprocess_sentence(self, sentence):
        step1 = sentence.lower()
        step2 = self._clean_urls(step1)
        step3 = self._tokenizer.tokenize(step2)
        step4 = self._remove_stop_words(step3)
        step5 = self._apply_noun_lemmatizer(step4)
        step6 = self._apply_verb_lemmatizer(step5)
        return step6

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

    def _get_numeric_labels(self, label_values):
        self._labels = np.unique(label_values)
        return list(map((lambda x: np.where(self._labels == x)[0][0]), label_values))

    def _get_specialized_vocabulary(self, tokenized_sentences, numeric_label_values):
        specialized_vocabulary = {}
        for index, tokenized_sentence in enumerate(tokenized_sentences):
            if numeric_label_values[index] not in specialized_vocabulary.keys():
                specialized_vocabulary[numeric_label_values[index]] = {}
            for token in tokenized_sentence:
                if token not in self._freq_words:
                    if token not in specialized_vocabulary[numeric_label_values[index]].keys():
                        specialized_vocabulary[numeric_label_values[index]][token] = 1
                    else:
                        specialized_vocabulary[numeric_label_values[index]][token] += 1

        for label in specialized_vocabulary.keys():
            class_dic = specialized_vocabulary[label]
            class_freqs = heapq.nlargest(self._frequent_words_per_label, class_dic, key=class_dic.get)
            self._freq_words += class_freqs

    def get_vocabulary(self):
        return np.array(self._sorted_vocabulary)

    def get_labels(self):
        return self._labels

    def get_bow_matrix(self, sentences, label_values=None):
        tokenized_sentences = list(map(self._preprocess_sentence, sentences))
        self._freq_words = heapq.nlargest(self._frequent_words_amount, self._vocabulary, key=self._vocabulary.get)
        if label_values is not None:
            numeric_labels = self._get_numeric_labels(label_values)
            self._get_specialized_vocabulary(tokenized_sentences, numeric_labels)
        bow_reps = list(map(self._get_bow_representation, tokenized_sentences))
        print(self._sorted_vocabulary)
        print(len(self._sorted_vocabulary))
        return np.column_stack((bow_reps, numeric_labels)) if label_values else np.array(bow_reps)
