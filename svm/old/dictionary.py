# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:57:40 2019

@author: Yassir
"""
import numpy as np


class Dictionary:
    
    def __init__(self, sorted_labels, masked = False):
        # labels
        n = sorted_labels.shape[0]
        self._n_classes = n
        self._labels = np.copy(sorted_labels)

        
        # all words seen so far
        self._all_words_counts = {}
        
        # words counts per class 
        self._words_counts_per_class = np.array([dict() for i in range(n)])
        
        # all classes where the word appears
        self._word_classes = {}
        
        # Mask
        self._masked = masked
        self._masked_words = []
        
    # Gets sorted classes
    def get_classes(self):
        return np.copy(self._labels)
    
    # Update the dictionary
    def update(self, tokenized_sentence, label) :
        for word in tokenized_sentence:
            # Updating the global dictionary
            if word in self._all_words_counts:
                self._all_words_counts[word] += 1
            else:
                self._all_words_counts[word] = 1
            
            # Updating the classes where the word belongs
            dic = self._words_counts_per_class[label]
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
            
            # Adding 'label' to the word's class
            if word in self._word_classes:
                self._word_classes[word].append(label)
            else:
                self._word_classes[word] = [label]
    
    # Get number of times the word 'word' appears per class
    def get_word_count_per_class(self, word, label = -1):
        # Wrong class id
        if (label > self._n_classes):
            print('Invalid label id given: could not find word %s' %word)
            return 0
        # Masked word
        if (self._masked == True) and (word in self._masked_words):
            #print('Word %s is banned!' %word)
            return 0
        # All words
        if label == -1:
            if word not in self._all_words_counts:
                return 0
            else:
                return self._all_words_counts[word]
        else:
            dic = self._words_counts_per_class[label]
            if word in dic:
                return dic[word]
            else:
                return 0
    
    # Get total number of words per class
    def get_total_count_per_class(self, label):
        if (label < 0) or (label > self._n_classes):
            print('Invalid label id given: could not find label id %d' %label)
            return 0
        return len(self._words_counts_per_class[label])
        
    # Update the list of masked words
    def update_mask(self, words):
        for word in words:
            if word not in self._masked_words:
                self._masked_words.append(word)
    
    # Set to true to filter banned words
    def set_apply_mask(self, value = False):
        self._masked = value
        

    # Total number of words
    def get_global_count(self):
        return len(self._all_words_counts)
    
    # Prints out dictionary
    def dump_dictionary(self, filepath = './out.txt'):
        dic = self._all_words_counts
        with open(filepath, 'w',  encoding="utf-8") as file:
            for w in sorted(dic, key=dic.get, reverse=True):
                file.write('%s : %d\n' %(w, dic[w]))
    
    # Prints out dictionary of classes
    def dump_dictionary_per_classes(self, max_class, filepath = './out.txt'):
        if max_class < 0 or max_class > 20:
            max_class = 20
        with open(filepath, 'w',  encoding="utf-8") as file:
            for c in range(max_class):
                dic = self._words_counts_per_class[c]
                file.write('Class %s (%d words)\n' % (self._labels[c], len(dic)))
                for w in sorted(dic, key=dic.get, reverse=True):
                    file.write('%s : %d\n' %(w, dic[w]))
                file.write('\n')
        
    
    
    