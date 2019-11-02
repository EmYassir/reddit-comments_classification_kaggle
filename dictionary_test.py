import numpy as np
from dictionary.dictionary import Dictionary
from text.text_util import Text_Util

train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'

print('### Loading data...')
comments, labels = np.load(train_set_path, allow_pickle=True)
# preparing labels
sorted_labels = np.unique(labels)
# Have to transform labels into numericals
sorted_labels_Ids = {}
index = 0
for l in sorted_labels:
    sorted_labels_Ids[l] = index
    index += 1

print('### Processing labels...')
numerical_labels = np.zeros(len(labels))
for i, y in enumerate(labels):
    numerical_labels[i] = sorted_labels_Ids[y]

# dictionary
print('### Creating dictionary...')
dic = Dictionary(sorted_labels)

# Preprocessing text
print('### Preprocessing text...')
text_util = Text_Util()
X = text_util.get_preprocessed_data_1(np.array(comments))
y = numerical_labels.astype(int)

# Updating dictionary
print('### Updating dictionary...')
for i in range(len(X)):
    dic.update(X[i], y[i])

# tests
print('### Testing dictionary:')
print('Total number of words: %d' %dic.get_global_count())
print('Total number of unique words: %d' %dic.get_count_unique_words())
print('Dumping content of dictionary...')
dic.dump_dictionary()
print('Dumping content of dictionary classes...')
dic.dump_dictionary_per_classes(len(sorted_labels))
print('Total count per labels:')
for i, l in enumerate(sorted_labels):
    print('Label %s: %d words' %(l, dic.get_total_count_per_class(i)))
print('Dumping 1000 top words of each label...')
for i in range(len(sorted_labels)): 
    dic.dump_top_words_per_label(1000, i)
print('Dumping 1000 bottom words of each label...')
for i in range(len(sorted_labels)): 
    dic.dump_bottom_words_per_label(1000, i)
print('Dumping all words common to all labels...')
dic.dump_words_common_to_labels(10000000, list(range(len(sorted_labels))))
    




