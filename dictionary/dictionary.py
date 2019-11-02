# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:57:40 2019

@author: Yassir
"""
import numpy as np


class Dictionary:
    
    def __init__(self, labels):
        
        # labels
        self._labels = np.array(labels)
        self._labels_Ids = np.arange(len(labels))

        # all words seen so far
        self._global_count = 0
        self._all_words_counts = {}
        
        # words counts per class 
        self._words_counts_per_class = np.array([dict() for i in range(len(labels))])
        
        # all classes where the word appears
        self._word_classes = {}

        
        
    # Gets sorted classes
    def get_labels(self):
        return np.copy(self._labels)
    
    # Update the dictionary
    def update_tokenized(self, tokenized_sentence, label) :
        for word in tokenized_sentence:
            # Updating the global dictionary
            if word in self._all_words_counts:
                self._all_words_counts[word] += 1
            else:
                self._all_words_counts[word] = 1
            # Update the global count 
            self._global_count += 1
            
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
    
     # Update the dictionary with non tokenized sentence
    def update_sentence(self, sentence, label) :
        tokenized_sentence = sentence.split(' ')
        self.update_tokenized(tokenized_sentence, label)
    
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
        if (label < 0) or (label >= len(self._labels)):
            print('Invalid label id given: could not find label id %d' %label)
            return 0
        return len(self._words_counts_per_class[label])

    # Total number of words
    def get_global_count(self):
        return self._global_count

    # Total number of different words words
    def get_count_unique_words(self):
        return len(self._all_words_counts)
    
    # Returns list of n most popular words per class
    def get_top_words_per_label(self, n, label = -1):
        if n <= 0:
            print('n should be positive')
            return
        dic = None
        # label == -1 means we re returning top words over all corpus
        if label  == -1:
            dic = self._all_words_counts
        else :
            dic = self._words_counts_per_class[label]
        
        sorted_list = sorted(dic, key=dic.get, reverse=True)
        return sorted_list[:min(len(sorted_list),n)]
    
    # Returns list of n least popular words per class
    def get_bottom_words_per_label(self, n, label = -1):
        dic = None
        # label == -1 means we re returning top words over all corpus
        if label  == -1:
            dic = self._all_words_counts
        else :
            dic = self._words_counts_per_class[label]
        
        sorted_list = sorted(dic, key=dic.get, reverse=False)
        return sorted_list[:min(len(sorted_list), n)]
    
        
    # Returns list of n common words between list of classes passed as parameter
    def get_words_common_to_labels(self, n, labels = []):
        res = []
        if labels  == []:
            print('Empty list of labels!')
            return res
        
        # Quadratic cost
        dic = self._all_words_counts
        count = 0
        sorted_global_list = sorted(self._all_words_counts, key=dic.get, reverse=True)
        for word in sorted_global_list:
            is_word_common = True
            for l in labels:
                if word not in self._words_counts_per_class[l]:
                    is_word_common = False
                    break
            # Adding word to the list
            if (is_word_common == True):
                res.append(word)
                count += 1
            # Checking if max is reached
            if count == n:
                break
        # End
        return res
    
       # Prints out dictionary
    def dump_dictionary(self, filepath = './dump/total_words.txt'):
        dic = self._all_words_counts
        with open(filepath, 'w',  encoding="utf-8") as file:
            for w in sorted(dic, key=dic.get, reverse=True):
                file.write('%s : %d\n' %(w, dic[w]))
    
    # Prints out dictionary of classes
    def dump_dictionary_per_classes(self, max_class, filepath = './dump/words_per_classes.txt'):
        if max_class < 0 or max_class > 20:
            max_class = 20
        with open(filepath, 'w',  encoding="utf-8") as file:
            for c in range(max_class):
                dic = self._words_counts_per_class[c]
                file.write('Class %s (%d words)\n' % (self._labels[c], len(dic)))
                for w in sorted(dic, key=dic.get, reverse=True):
                    file.write('%s : %d\n' %(w, dic[w]))
                file.write('\n')
    
        # Returns list of n most popular words per class
    #def get_top_words_per_label(self, n, label = -1):
    def dump_top_words_per_label(self, n, label = -1):
        filepath = ''
        if label == -1:
            filepath = 'all'
        elif label >-1 :
            filepath = self._labels[label]
        else:
            print('Cannot find label %d' %label)
        
        filepath = './dump/top_words_per_label_'+ str(filepath) + '.txt'
        list_words =  self.get_top_words_per_label(n, label)
        with open(filepath, 'w',  encoding="utf-8") as file:
            file.write('Label %s (%d top words):\n' % (self._labels[label], n))
            for w in list_words:
                file.write('%s\n, ' %w)
            file.write('\n')
        
        
    #def get_bottom_words_per_label(self, n, label = -1):
    def dump_bottom_words_per_label(self, n, label = -1):
        filepath = ''
        if label == -1:
            filepath = 'all'
        elif label >-1 :
            filepath = self._labels[label]
        else:
            print('Cannot find label %d' %label)
        
        filepath = './dump/bottom_words_per_label_'+ str(filepath) + '.txt'
        list_words =  self.get_bottom_words_per_label(n, label)
        with open(filepath, 'w',  encoding="utf-8") as file:
            file.write('Label %s (%d bottom words):\n' % (self._labels[label], n))
            for w in list_words:
                file.write('%s\n' %w)
            file.write('\n')
            
    #def get_words_common_to_labels(self, n, labels = []):
    def dump_words_common_to_labels(self, n, labels = []):
        filepath = ''
        if labels  == []:
            print('Empty list of labels!')
            return 
        else:
            for l in labels:
               filepath = filepath + '_' + str(self._labels[l])
        
        filepath = './dump/common_words_btw_labels'+ str(filepath) + '.txt'
        list_words =  self.get_words_common_to_labels(n, labels)
        with open(filepath, 'w',  encoding="utf-8") as file:
            file.write('Common words sharing labels: ')
            for l in labels:
                file.write('%s ' %self._labels[l])
            file.write('\n Total of %d words \n' %len(list_words))
            for w in list_words:
                file.write('%s, ' %w)
            file.write('\n')
        
        
        
    
    