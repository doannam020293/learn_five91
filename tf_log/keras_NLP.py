# texts = []  # list of text samples
# labels_index = {}  # dictionary mapping label name to numeric id
# labels = []  # list of label ids
# for name in sorted(os.listdir(TEXT_DATA_DIR)):
#     path = os.path.join(TEXT_DATA_DIR, name)
#     if os.path.isdir(path):
#         label_id = len(labels_index)
#         labels_index[name] = label_id
#         for fname in sorted(os.listdir(path)):
#             if fname.isdigit():
#                 fpath = os.path.join(path, fname)
#                 if sys.version_info < (3,):
#                     f = open(fpath)
#                 else:
#                     f = open(fpath, encoding='latin-1')
#                 t = f.read()
#                 i = t.find('\n\n')  # skip header
#                 if 0 < i:
#                     t = t[i:]
#                 texts.append(t)
#                 f.close()
#                 labels.append(label_id)
#
# print('Found %s texts.' % len(texts))
#
#
#
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
#
# tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
#
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
#
# data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#
# labels = to_categorical(np.asarray(labels))
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)
#
# # split the data into a training set and a validation set
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
#
# x_train = data[:-nb_validation_samples]
# y_train = labels[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]