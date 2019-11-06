import numpy as np
from dictionary.dictionary import Dictionary
from text.text_util import Text_Util

TOKENIZED = 0

train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'

print('### Loading data...')
comments, labels = np.load(train_set_path, allow_pickle=True)
# preparing labels
sorted_labels = np.unique(labels)

# dictionary
print('### Creating dictionary...')
dic = Dictionary(sorted_labels)

# Preprocessing text
print('### Preprocessing text...')
text_util = Text_Util()
X = None
if TOKENIZED == 1:
    X = text_util.get_preprocessed_tokenized_sentences(np.array(comments))
else:
    X = text_util.get_preprocessed_sentences(np.array(comments))

print('Total words in the corpus before cleanup: %d' 
      %(text_util.get_number_scanned_words()))
      
# Updating dictionary
print('### Updating dictionary...')
for i in range(len(X)):
    if TOKENIZED == 1:
        dic.update_tokenized(X[i], labels[i])
    else:
        dic.update_sentence(X[i], labels[i])

# tests
print('### Testing dictionary:')
print('Total number of words: %d' %dic.get_global_count())
print('Total number of unique words: %d' %dic.get_count_unique_words())

print('Dumping content of dictionary...')
dic.dump_dictionary()
print('Dumping content of dictionary labels...')
dic.dump_dictionary_labels()
print('Total count per labels:')
for i, l in enumerate(sorted_labels):
    print('Label %s: %d words' %(l, dic.get_total_count_given_label(l)))
print('Dumping 1000 top words of each label...')
for l in sorted_labels: 
    dic.dump_n_top_words_given_label(l, 1000)
print('Dumping 1000 bottom words of each label...')
for l in sorted_labels: 
    dic.dump_n_bottom_words_given_label(l, 1000)
print('Dumping all words common to all labels...')
dic.dump_n_common_words_to_labels()
print('Dumping all words unique to each label...')
dic.dump_n_unique_words_to_labels()
print('Dumping all common to \'AskReddit\' and \'Music\'')
dic.dump_n_common_words_to_labels(labels = ['AskReddit', 'Music'])



    




