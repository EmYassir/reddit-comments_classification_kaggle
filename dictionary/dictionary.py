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

        # all words seen so far
        self._global_count = 0
        self._all_words_counts = {}
        
        # words counts given a label 
        self._words_counts_given_label = {}
        for l in labels:
            self._words_counts_given_label[l] = {}
        
        # all labels where the word appears
        self._word_labels = {}

        
        
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
            dic = self._words_counts_given_label[label]
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
            
            # Adding 'label' to the word's class
            if word in self._word_labels:
                self._word_labels[word].append(label)
            else:
                self._word_labels[word] = [label]
    
    
    # Update the dictionary with non tokenized sentence
    def update_sentence(self, sentence, label) :
        tokenized_sentence = sentence.split(' ')
        self.update_tokenized(tokenized_sentence, label)
    
    
    # Get number of times the word 'word' appears per label
    def get_word_count_given_label(self, word, label):
        # Wrong class id
        if label not in self._labels:
            print('Unknown label: %s' %label)
            return -1
        dic = self._words_counts_given_label[label]
        if word in dic:
            return dic[word]
        else:
            print('Word %s not found under label: %s' %(word, label))
            return -1
    
    # Get total number of words per class
    def get_total_count_given_label(self, label):
        if label not in self._labels:
            print('Invalid label id given: could not find label id %d' %label)
            return 0
        return len(self._words_counts_given_label[label])

    # Total number of words
    def get_global_count(self):
        return self._global_count

    # Total number of different words words
    def get_count_unique_words(self):
        return len(self._all_words_counts)
    
    
    # Returns list of n most popular words per class (n <= 0 means no limit)
    def get_n_top_words_given_label(self, label, n = 0):
        res = []
        if label not in self._labels:
            print('Unknown label: %s' %label)
            return res
        
        dic = self._words_counts_given_label[label]
        res = sorted(dic, key=dic.get, reverse=True)
        if n <= 0:
            return res
        else :
            return res[:min(len(res),n)]
    
    # Returns list of n least popular words per class (n <= 0 means no limit)
    def get_n_bottom_words_given_label(self, label, n = 0):
        res = []
        if label not in self._labels:
            print('Unknown label: %s' %label)
            return res   
        dic = self._words_counts_given_label[label]
        res = sorted(dic, key=dic.get, reverse=False)
        if n <= 0:
            return res
        else :
            return res[:min(len(res),n)]
    
        
    # Returns list of n common words between list of labels passed as parameter
    # Empty list means that all labels will be considered
    # n <= 0 means no limit
    def get_n_words_common_to_labels(self, labels = [], n = 0):
        res = []
        
        # Total complexity cost : Quadratic
        dic = self._all_words_counts
        sorted_global_list = sorted(self._all_words_counts, key=dic.get, reverse=True)
        
        # Label list to scan
        labels_to_scan = []
        if len(labels) == 0:
            labels_to_scan = self._labels
        else:
            for l in labels:
                if l not in self._labels:
                    continue
                else:
                    labels_to_scan.append(l)
        # updating limit to scan
        limit = n
        if limit <= 0:
            limit = self._global_count
        
        count = 0
        print('Scanning %d words' %len(sorted_global_list))
        for word in sorted_global_list:
            is_word_common = True
            for l in labels_to_scan:
                if word not in self._words_counts_given_label[l]:
                    is_word_common = False
                    break
            # Adding word to the list
            if (is_word_common == True):
                res.append(word)
                count += 1
            # Checking if max is reached
            if count >= limit:
                break
        # End
        return res
    
    # Returns list of n words unique to given label
    # n <= 0 means no limit
    def get_n_words_unique_to_label(self, label, n = 0):
        res = []
        if label not in self._labels:
            print('Unknown label: %s' %label)
            return res   
        # Getting the list of words belonging to the given label
        dic = self._words_counts_given_label[label]
        aux_list = sorted(dic, key=dic.get, reverse=True)
        for elem in aux_list:
            is_unique = True
            for l in self._labels:
                if l == label:
                    continue
                if elem in self._words_counts_given_label[l]:
                    is_unique = False
                    break
            if is_unique:
                res.append(elem)
        
        limit = len(res)
        if (n > 0):
            limit = min(n, limit)
        return res[:limit]           
        
        
    
    # Prints out dictionary
    def dump_dictionary(self, filepath = './dump/all_words.txt'):
        dic = self._all_words_counts
        with open(filepath, 'w',  encoding="utf-8") as file:
            for w in sorted(dic, key=dic.get, reverse=True):
                file.write('%s : %d\n' %(w, dic[w]))
    
    
    # Prints out content of each label
    def dump_dictionary_labels(self, labels = [], filepath = './dump/dic_labels.txt'):
        # Label list to scan
        labels_to_scan = []
        if len(labels) == 0:
            labels_to_scan = self._labels
        else:
            for l in labels:
                if label not in self._labels:
                    continue
                else:
                    labels_to_scan.append(l)
        # Writing on file
        with open(filepath, 'w',  encoding="utf-8") as file:
            for l in labels_to_scan:
                dic = self._words_counts_given_label[l]
                file.write('Label %s (%d words)\n' % (l, len(dic)))
                for w in sorted(dic, key=dic.get, reverse=True):
                    file.write('%s : %d\n' %(w, dic[w]))
                file.write('\n')
    
    # Prints out n top words of the given label
    def dump_n_top_words_given_label(self, label, n = 0):
        # Invalid label
        if label not in self._labels:
            print ('Unknown label %s' %label)
            return

        
        list_words = self.get_n_top_words_given_label(label, n)
        # 'Update' n
        if n <= 0:
            n = len(list_words)
        # Write on file
        filepath = './dump/' + str(n) + '_top_words_given_label_'+ str(label) + '.txt'
        with open(filepath, 'w',  encoding="utf-8") as file:
            file.write('Label %s (%d top words):\n' % (label, n))
            for w in list_words:
                file.write('%s\n' %w)
            file.write('\n')
        
    # Prints out n last words of the given label
    def dump_n_bottom_words_given_label(self, label, n = 0):
        # Invalid label
        if label not in self._labels:
            print ('Unknown label %s' %label)
            return
        # Get list of words
        list_words =  self.get_n_bottom_words_given_label(label, n)
        # 'Update' n
        if n <= 0:
            n = len(list_words)
        # Write on file
        filepath = './dump/' + str(n) + '_bottom_words_given_label_'+ str(label) + '.txt'
        with open(filepath, 'w',  encoding="utf-8") as file:
            file.write('Label %s (%d bottom words):\n' % (label, n))
            for w in list_words:
                file.write('%s\n' %w)
            file.write('\n')
            
    # Prints out n common words to labels
    def dump_n_common_words_to_labels(self, labels = [], n = 0):
        # Label list to scan
        labels_to_scan = []
        if len(labels) == 0:
            labels_to_scan = self._labels
        else:
            for l in labels:
                if l not in self._labels:
                    continue
                else:
                    labels_to_scan.append(l)
        
        # Get list of words
        list_words =  self.get_n_words_common_to_labels(labels_to_scan, n)
        
        # 'Update' n
        if n <= 0:
            n = len(list_words)
        
        # file name
        filepath = ''
        if len(labels_to_scan) == len(self._labels):
            filepath = './dump/' + str(n) + '_common_words_all_labels.txt'
        else:
            filepath = './dump/' + str(n) + '_common_words'
            for l in labels_to_scan:
                filepath = filepath + '_' + str(l)
            filepath += '_labels.txt'
        
        with open(filepath, 'w',  encoding="utf-8") as file:
            file.write('Common words sharing labels: ')
            for l in labels:
                file.write('%s ' %l)
            file.write('\n Total of %d words \n' %len(list_words))
            for w in list_words:
                file.write('%s, ' %w)
            file.write('\n')
        
    # Prints out n unique words to labels
    def dump_n_unique_words_to_labels(self, labels = [], n = 0):
        # Label list to scan
        labels_to_scan = []
        if len(labels) == 0:
            labels_to_scan = self._labels
        else:
            for l in labels:
                if l not in self._labels:
                    continue
                else:
                    labels_to_scan.append(l)
        
        # Get array of lists of words
        list_words = []
        for i, l in enumerate(labels_to_scan):
            list_words.append(self.get_n_words_unique_to_label(l, n))
        
        # file name
        filepath = ''
        
        # 'Update' n for printing
        if n <= 0:
            filepath = 'all'
        else:
            filepath = str(n)
        
        if len(labels_to_scan) == len(self._labels):
            filepath = './dump/' + filepath + '_unique_words_all_labels.txt'
        else:
            filepath = './dump/' + filepath + '_unique_words'
            for l in labels_to_scan:
                filepath = filepath + '_' + str(l)
            filepath += '_labels.txt'
        
        with open(filepath, 'w',  encoding="utf-8") as file:
            file.write('Unique words per labels: \n')
            for i,l in enumerate(labels_to_scan):
                file.write('=> Label \'%s\' (total of %d unique words):\n' %(l, len(list_words[i])))
                for w in list_words[i]:
                    file.write('%s\n' %w)
                file.write('\r\n')
        
        
    
    