import numpy as np
from keras.layers import Conv1D, MaxPool1D, Dense, Embedding, Flatten
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import  Input
import os
import sys
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.losses import sparse_categorical_crossentropy

pre_train_file = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\nlp\glove.6B.50d.txt'
TEXT_DATA_DIR = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\nlp\20_newsgroup'
nb_words = 20000 # số lượng từ tối đa có thể xuất hiện khi convert 1 document ra 1 sequence
nb_max_len_sentence = 1000
percent_test = 0.1
nb_dimension =1
embed_vector = {}


with open(pre_train_file,'rb') as file:
    for line in file:
        line_split  = line.split()
        word = line_split[0].decode('utf-8')
        vector = np.asarray(line_split[1:],dtype='float32')
        embed_vector[word] = vector






# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)


print('Found %s texts.' % len(texts))



tokenizer = Tokenizer(num_words=nb_words)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(texts)

data  = pad_sequences(sequences=sequences,maxlen=nb_max_len_sentence)
label_one_hot = to_categorical(labels)
# split data to train test
shuffle_array =  np.array(range(data.shape[0]))
np.random.shuffle(shuffle_array)
number_test = int(data.shape[0]*percent_test)


X_train, X_test = data[shuffle_array[number_test:]],data[shuffle_array[:number_test]]
y_train, y_test = label_one_hot[number_test:],label_one_hot[:number_test]


nb_vocabulary = len(word_index)
embed_layer = np.zeros(shape=(nb_vocabulary,50))
for word, i in word_index.items():
    # if i < nb_words:
    embedding_vector = embed_vector.get(word)

    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embed_layer[i,:] = embedding_vector





# doạn code này xóa đi khi chạy thật
# X_train = 0
# y_train = 0
# nb_vocabulary =10000
# embed_layer = np.zeros(shape=(nb_vocabulary,50))

#########################
# xay model

input_layer = Input(
    shape=(nb_max_len_sentence,),
    dtype= 'int32'
)
embed_layer1 = Embedding(
    input_dim= nb_vocabulary,
    output_dim=50,
    weights=[embed_layer],
    input_length=nb_max_len_sentence,
    trainable=False
)
size_filter = 128
x = embed_layer1(input_layer)
x = Conv1D(filters=size_filter,kernel_size=5)(x)
x = MaxPool1D(pool_size=5)(x)
x = Conv1D(filters=size_filter,kernel_size=5)(x)
x = MaxPool1D(pool_size=5)(x)
x = Flatten()(x)
x = Dense(
    units=128,
    activation='relu',
)(x)
x = Dense(
    units=20,
    activation='softmax',
    name='softmax'
)(x)


model = Model(inputs=input_layer,outputs=x)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy', #loss= sparse_categorical_crossentropy(), #
)

model.fit(
    x= X_train,
    y = y_train,
    batch_size=32,
    epochs=5
)

model.save('nam_model_nlp.h5')