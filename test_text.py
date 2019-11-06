import numpy as np
from text.text_util import Text_Util


train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'

def dump_text(data, filepath):
    with open(filepath, 'w',  encoding="utf-8") as file:
        for comment in data:
            file.write('%s\n\n' %comment)

print('### Loading data...')
comments, labels = np.load(train_set_path, allow_pickle=True)
print('### Dumping raw data...')
dump_text(comments, './output/raw_data.txt')

# preparing labels
sorted_labels = np.unique(labels)


# Preprocessing text
print('### Preprocessing text...')
text_util = Text_Util()
X = None
if TOKENIZED == 1:
    X = text_util.get_preprocessed_tokenized_sentences(np.array(comments))
else:
    X = text_util.get_preprocessed_sentences(np.array(comments))
    
    
print('### Dumping cleaned data...')    
dump_text(X, './output/cleaned_data.txt')




    




