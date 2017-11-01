# # # # from keras.layers import Input, Dense
# # # # from keras.models import Model
# # # #
# # # # # This returns a tensor
# # # # inputs = Input(shape=(784,))
# # # #
# # # # # a layer instance is callable on a tensor, and returns a tensor
# # # # x = Dense(64, activation='relu')(inputs)
# # # # x = Dense(64, activation='relu')(x)
# # # # predictions = Dense(10, activation='softmax')(x)
# # # #
# # # # # This creates a model that includes
# # # # # the Input layer and three Dense layers
# # # # model = Model(inputs=inputs, outputs=predictions)
# # # # model.compile(optimizer='rmsprop',
# # # #               loss='categorical_crossentropy',
# # # #               metrics=['accuracy'])
# # # # # model.fit(data, labels)  # starts training
# # # # x = Input(shape=(784,))
# # # # # This works, and returns the 10-way softmax we defined above.
# # # # y = model(x)
# # # #
# # # #
# # # # from keras.applications.vgg16 import VGG16
# # # # from keras.models import Sequential
# # # # from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# # # # from keras.layers import Conv2D, MaxPool2D
# # # # from keras.layers import Activation, Dropout, Flatten, Dense
# # # # from keras.optimizers import SGD
# # # #
# # # #
# # # # import  os
# # # # import numpy as np
# # # #
# # # #
# # # # def move_file():
# # # #     path = r'C:\nam\work\data\train'
# # # #     cat_path = r'C:\nam\work\data\cats'
# # # #     dog_path = r'C:\nam\work\data\dogs'
# # # #
# # # #     for root, dirs, files in os.walk(path):
# # # #         for file in files:
# # # #             full_file = os.path.join(root,file)
# # # #             if 'cat' in file:
# # # #                 new_file = os.path.join(cat_path , file)
# # # #                 os.rename(full_file,new_file)
# # # #             if 'dog' in file:
# # # #                 new_file = os.path.join(dog_path, file)
# # # #                 os.rename(full_file, new_file)
# # # # def test_data_augmentation():
# # # #     img = load_img('data/train/cats/cat.2.jpg')
# # # #     y = img_to_array(img)
# # # #     y.shape
# # # #
# # # #     img = load_img('data/train/cats/cat.1.jpg')
# # # #     img = load_img(r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\preview\cat_0_431.jpeg')
# # # #     x = img_to_array(img)
# # # #     x = x.reshape((1,)+x.shape)
# # # #
# # # #     i = 0
# # # #     for batch in datagen.flow(x,batch_size=1,
# # # #                               save_to_dir=r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\preview',
# # # #                               save_prefix='cat',
# # # #                               save_format='jpeg'):
# # # #         i +=1
# # # #         if i>20:
# # # #             break
# # # #
# # # # img_width, img_height = 150, 150
# # # # top_model_weights_path = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\data_input\learn\bottleneck_fc_model.h5'
# # # # train_data_dir = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\train'
# # # # validation_data_dir = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\validation'
# # # # nb_train_samples = 1032 * 2
# # # # nb_validation_samples = 645*2
# # # # epochs = 50
# # # # batch_size = 16
# # # #
# # # #
# # # # model  = VGG16(weights='imagenet',include_top=True)
# # # # datagen = ImageDataGenerator(
# # # #     rotation_range=40,
# # # #     width_shift_range=0.2,
# # # #     height_shift_range=0.2,
# # # #     rescale=1./255,
# # # #     shear_range=0.2,
# # # #     zoom_range=0.2,
# # # #     horizontal_flip=True,
# # # #     fill_mode='nearest'
# # # # )
# # # # # cho dù ảnh vào có kích thước ntn đi thì qua datagen đều ra output có dnag (150,150)
# # # # generator = datagen.flow_from_directory(
# # # #     directory= r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\train',
# # # #     target_size=(150,150),
# # # #     batch_size=batch_size,
# # # #     class_mode=None,
# # # #     shuffle=False
# # # # )
# # # #
# # # # bottleneck_featture_train = model.predict_generator(generator=generator,steps=2000)
# # # #
# # # # np.save(open('bottleneck_featture_train.npy','w'),bottleneck_featture_train)
# # # #
# # # # generator = datagen.flow_from_directory(
# # # #     directory=r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\validation',
# # # #     target_size=(150, 150),
# # # #     batch_size=batch_size,
# # # #     class_mode=None,
# # # #     shuffle=False
# # # # )
# # # # bottleneck_featture_validation = model.predict_generator(generator=generator,steps=800)
# # # # np.save(open('bottleneck_featture_validation.npy','w'),bottleneck_featture_validation )
# # # #
# # # #
# # # # train_data = np.load(open('bottleneck_featture_train.npy'))
# # # # train_labels = np.array([0]*1032 + [1]*1032)
# # # #
# # # # validation_data = np.load(open('bottleneck_featture_validation.npy'))
# # # # train_labels = np.array([0]*645 + [1]*645)
# # # #
# # # #
# # # # top_model = Sequential()
# # # # top_model.add(Flatten(input_shape=model.output_shape[1:]))
# # # # top_model.add(Dense(256,activation='relu'))
# # # # top_model.add(Dropout(0.5))
# # # # top_model.add(Dense(1,activation='sigmoid'))
# # # #
# # # # # k hiểu cái weight này chỉ weeight nào, do mỗi model có cấu trúc khác nhau, vì thế số lượng weight cũng khác nhau.
# # # # top_model.load_weights(top_model_weights_path)
# # # #
# # # # model.add(top_model)
# # # #
# # # # for layer in model.layers[:25]:
# # # #     layer.trainabel  = False
# # # #
# # # # model.compile(loss='binary_crossentropy',
# # # #               optimizer=SGD(lr=1e-4, momentum=0.9),
# # # #               metrics=['accuracy'])
# # # #
# # # # batch_size = 16
# # # #
# # # #
# # # # train_datagen = ImageDataGenerator(
# # # #     rescale=1./255,
# # # #     shear_range=0.2,
# # # #     zoom_range=0.2,
# # # #     horizontal_flip=True
# # # # )
# # # #
# # # # test_datagen = ImageDataGenerator(rescale=1./255)
# # # # train_generator = train_datagen.flow_from_directory(
# # # #     train_data_dir,
# # # #     target_size=(img_height, img_width),
# # # #     batch_size=batch_size,class_mode='binary')
# # # #
# # # # validation_generator = test_datagen.flow_from_directory(
# # # #         validation_data_dir,
# # # #         target_size=(img_height, img_width),
# # # #         batch_size=batch_size,
# # # #         class_mode='binary')
# # # #
# # # # model.fit_generator(
# # # #     generator= train_generator,
# # # #     steps_per_epoch= nb_train_samples//batch_size,
# # # #     epochs=epochs,
# # # #     validation_data=validation_generator,
# # # #     validation_steps=nb_validation_samples//batch_size
# # # # )
# # #
# # # import numpy as np
# # # GLOVE_DIR = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\nlp\word2vec fasttext\wiki.vi.vec'
# # # embeddings_index = {}
# # # f = open(GLOVE_DIR,'rb')
# # # for line in f:
# # #     values = line.split()
# # #     # word = string(values[0])
# # #     word = values[0].decode("utf-8")
# # #     coefs = np.asarray(values[1:], dtype='float32')
# # #     embeddings_index[word] = coefs
# # # f.close()
# # #
# # #
# # # nb_words = 3
# # # tokenizer = Tokenizer(nb_words=nb_words)
# # # tokenizer.fit_on_texts(["The sun is shining in June!","September is grey.","Life is beautiful in August.","I like it","This and other things?"])
# # # print(tokenizer.word_index)
# # #
# # # a = tokenizer.texts_to_sequences([" the in is beautiful and I like it!"])
# # #
# # #
# # # a = tokenizer.texts_to_matrix(["in "])
# # # tokenizer.texts_to_matrix(["the "])
# # # tokenizer.texts_to_sequences(["is "])
# # # data  = pad_sequences(sequences=a,maxlen=5)
# # # pad_sequences(sequences=a,maxlen=3)
# # import pandas as pd
# # from pandas import read_csv
# # from datetime import datetime
# # from pandas import DataFrame
# # from sklearn.preprocessing.data import MinMaxScaler
# # from sklearn.preprocessing.label import LabelEncoder
# # from matplotlib import pyplot
# # from keras.models import Sequential, Model
# # from keras.layers import LSTM, Dense
# # #
# # #
# # # # load data
# # # def parse(x):
# # # 	return datetime.strptime(x, '%Y %m %d %H')
# # # dataset = read_csv(r'C:\Users\Windows 10 TIMT\Downloads\raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
# # # # dataset = read_csv(r'C:\Users\Windows 10 TIMT\Downloads\raw.csv',  parse_dates = ['year', 'month', 'day', 'hour'], index_col=0)
# # # # dataset.drop('No', axis=1, inplace=True)
# # # # manually specify column names
# # # dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# # # dataset.index.name = 'date'
# # # # mark all NA values with 0
# # # dataset['pollution'].fillna(0, inplace=True)
# # # # drop the first 24 hours
# # # dataset = dataset[24:]
# # # # summarize first 5 rows
# # # print(dataset.head(5))
# # # # save to file
# # # dataset.to_csv('pollution.csv')
# # #
# # #
# # # from pandas import read_csv
# # # from matplotlib import pyplot
# # # # load dataset
# # # dataset = read_csv('pollution.csv', header=0, index_col=0)
# # # values = dataset.values
# # # # specify columns to plot
# # # groups = [0, 1, 2, 3, 5, 6, 7]
# # # i = 1
# # # # plot each column
# # # pyplot.figure()
# # # for group in groups:
# # # 	pyplot.subplot(len(groups), 1, i)
# # # 	pyplot.plot(values[:, group])
# # # 	pyplot.title(dataset.columns[group], y=0.5, loc='right')
# # # 	i += 1
# # # pyplot.show()
# # #
# #
# # # convert series to supervised learning
# # def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
# # 	n_vars = 1 if type(data) is list else data.shape[1]
# # 	df = DataFrame(data)
# # 	cols, names = list(), list()
# # 	# input sequence (t-n, ... t-1)
# # 	for i in range(n_in, 0, -1):
# # 		cols.append(df.shift(i))
# # 		names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
# # 	# forecast sequence (t, t+1, ... t+n)
# # 	for i in range(0, n_out):
# # 		cols.append(df.shift(-i))
# # 		if i == 0:
# # 			names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
# # 		else:
# # 			names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
# # 	# put it all together
# # 	agg = pd.concat(cols, axis=1)
# # 	agg.columns = names
# # 	# drop rows with NaN values
# # 	if dropnan:
# # 		agg.dropna(inplace=True)
# # 	return agg
# #
# #
# # # load dataset
# # dataset = read_csv('pollution.csv', header=0, index_col=0)
# # values = dataset.values
# # # integer encode direction
# # encoder = LabelEncoder()
# # values[:, 4] = encoder.fit_transform(values[:, 4])
# # # ensure all data is float
# # values = values.astype('float32')
# # # normalize features
# # scaler = MinMaxScaler(feature_range=(0, 1))
# # scaled = scaler.fit_transform(values)
# # # frame as supervised learning
# # reframed = series_to_supervised(scaled, 1, 1)
# # # drop columns we don't want to predict
# # reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# # print(reframed.head())
# #
# #
# #
# # # split into train and test sets
# # values = reframed.values
# # n_train_hours = 365 * 24
# # train = values[:n_train_hours, :]
# # test = values[n_train_hours:, :]
# # # split into input and outputs
# # train_X, train_y = train[:, :-1], train[:, -1]
# # test_X, test_y = test[:, :-1], test[:, -1]
# # # reshape input to be 3D [samples, timesteps, features]
# # train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# # test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# #
# # # design network
# # model = Sequential()
# # model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# # model.add(Dense(1))
# # model.compile(loss='mae', optimizer='adam')
# # # fit network
# # history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# # # plot history
# # pyplot.plot(history.history['loss'], label='train')
# # pyplot.plot(history.history['val_loss'], label='test')
# # pyplot.legend()
# # pyplot.show()
# #
# # To support both python 2 and python 3
# from __future__ import division, print_function, unicode_literals
# import math
# import numpy as np
# import matplotlib.pyplot as plt
#
# N = 100 # number of points per class
# d0 = 2 # dimensionality
# C = 3 # number of classes
# X = np.zeros((d0, N*C)) # data matrix (each row = single example)
# y = np.zeros(N*C, dtype='uint8') # class labels
#
# for j in range(C):
#   ix = range(N*j,N*(j+1))
#   r = np.linspace(0.0,1,N) # radius
#   t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
#   X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
#   y[ix] = j
# # lets visualize the data:
# # plt.scatter(X[:N, 0], X[:N, 1], c=y[:N], s=40, cmap=plt.cm.Spectral)
#
# def softmax(V):
#     e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
#     Z = e_V / e_V.sum(axis = 0)
#     return Z
#
# ## One-hot coding
# from scipy import sparse
# def convert_labels(y, C = 3):
#     Y = sparse.coo_matrix((np.ones_like(y),
#         (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
#     return Y
#
# # cost or loss function
# def cost(Y, Yhat):
#     return -np.sum(Y*np.log(Yhat))/Y.shape[1]
#
# d0 = 2
# d1 = h = 100 # size of hidden layer
# d2 = C = 3
# # initialize parameters randomly
# W1 = 0.01*np.random.randn(d0, d1)
# b1 = np.zeros((d1, 1))
# W2 = 0.01*np.random.randn(d1, d2)
# b2 = np.zeros((d2, 1))
#
# Y = convert_labels(y, C)
# N = X.shape[1]
# eta = 1 # learning rate
# for i in range(10000):
#     ## Feedforward
#     Z1 = np.dot(W1.T, X) + b1
#     A1 = np.maximum(Z1, 0)
#     Z2 = np.dot(W2.T, A1) + b2
#     Yhat = softmax(Z2)
#
#     # print loss after each 1000 iterations
#     if i %1000 == 0:
#         # compute the loss: average cross-entropy loss
#         loss = cost(Y, Yhat)
#         print("iter %d, loss: %f" %(i, loss))
#
#     # backpropagation
#     E2 = (Yhat - Y )/N
#     dW2 = np.dot(A1, E2.T)
#     db2 = np.sum(E2, axis = 1, keepdims = True)
#     E1 = np.dot(W2, E2)
#     E1[Z1 <= 0] = 0 # gradient of ReLU
#     dW1 = np.dot(X, E1.T)
#     db1 = np.sum(E1, axis = 1, keepdims = True)
#
#     # Gradient Descent update
#     W1 += -eta*dW1
#     b1 += -eta*db1
#     W2 += -eta*dW2
#     b2 += -eta*db2
import  tensorflow as tf
import  numpy as np
#
# a = tf.constant(1,name='constant')
# x = tf.Variable(1,)
# sign_op2 = tf.assign_sub(x,10)
# sign_op1 = tf.assign(x,2)
#
#
# # x1 = tf.Variable(tf.random_normal([1,2]))
# # x2 = tf.Variable(x1.initial_value*2)
# init = tf.global_variables_initializer()
#
# # g= tf.Graph()
# # with tf.control_dependencies([init,sign_op2,sign_op1]):
# with tf.control_dependencies([sign_op2, sign_op1]):
# # with tf.control_dependencies([init]):
# 	sub_op = tf.assign_sub(x,20)
# # with tf.Session() as sess:
# # sess = tf.Session()
# sess =tf.InteractiveSession()
# # a = sess.run(sign_op)
# sess.run(init)
# a = sess.run(sub_op)
# # a= x2.eval()
# # print(x.eval())
# print(a)
#
# sess.close()
# a = tf.Variable(1,dtype=tf.float32)
# a_assign = tf.assign(a,2)
# b = 2*a**3
# c = b**2 + 2*b
# grad = tf.gradients(c,[b,a])
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
# 	sess.run(init) # returns 45
# 	# sess.run(a_assign) # returns 45
# 	x = sess.run(grad) # 34 816
# 	print(x)
# saver = tf.train.Saver() # defaults to saving all variables
# saver.save()

# def step_by_step_RNN():
# 	n_neuron = 5
# 	# n_output = 3
# 	n_input = 3
# 	X1 = tf.placeholder(tf.float32,[None,n_input])
# 	X2 = tf.placeholder(tf.float32,[None,n_input])
#
# 	Wx = tf.Variable(tf.random_normal([n_input,n_neuron]))
# 	Wh = tf.Variable(tf.random_normal([n_neuron,n_neuron]))
# 	# Wy = tf.Variable(tf.random_normal([n_neuron,n_output]))
# 	b1 = tf.Variable(tf.zeros(n_neuron))
# 	# b2 = tf.Variable(tf.zeros(n_output))
# 	h1 = tf.tanh(tf.matmul(X1,Wx) + b1)
# 	# Y1 = tf.tanh(tf.matmul(h1,Wy)+b2)
# 	h2 = tf.tanh(tf.matmul(X1,Wx)+tf.matmul(h1,Wh) + b1)
# 	# Y2 = tf.tanh(tf.matmul(h2,Wy)+b2)
#
# 	init = tf.global_variables_initializer()
# 	X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
# 	X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1
# 	with tf.Session() as sess:
# 		sess.run(init)
# 		y1_eval, y2_eval = sess.run([h1,h2],feed_dict={X1:X0_batch, X2:X1_batch})
# 		print(y1_eval, y2_eval )


#
# num_batches, batch_size, num_features = 1,1,1
# lstm_size = 1
# words_in_dataset = tf.placeholder(tf.float32, [num_batches, batch_size, num_features])
# lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size,)
# # Initial state of the LSTM memory.
# # hidden_state = tf.zeros([batch_size, lstm.state_size])
# # current_state = tf.zeros([batch_size, lstm.state_size])
# hidden_state = tf.zeros([batch_size, 1])
# current_state = tf.zeros([batch_size, 1])
# state = hidden_state, current_state
# probabilities = []
# loss = 0.0
# for current_batch_of_words in words_in_dataset:
# 	current_batch_of_words = 0
#     # The value of state is updated after processing each batch of words.
#     output, state = lstm(current_batch_of_words, state)
#
#     # The LSTM output can be used to make next word predictions
#     logits = tf.matmul(output, softmax_w) + softmax_b
#     probabilities.append(tf.nn.softmax(logits))
#     loss += loss_function(probabilities, target_words)
#
# words_in_dataset = []
# words_in_dataset[0] = ['The', 'The']
# words_in_dataset[1] = ['fox', 'fox']
# words_in_dataset[2] = ['is', 'jumped']
# words_in_dataset[3] = ['quick', 'high']
# num_batches = 4
# batch_size = 2
# time_steps = 5



n_neuron = 5
# n_output = 3
n_input = 3
n_step =2
batch_size =1
# sequence_length = tf.placeholder(tf.float32,shape=[None])
# X = tf.placeholder(tf.float32, [batch_size,n_step, n_input])
# output, state  = tf.nn.dynamic_rnn(cell, X,sequence_length=sequence_length,dtype=tf.float32)
#
# X_batch = np.array([
# 	[[0, 1, 2], [9, 8, 7]], # instance 0
# 	[[3, 4, 5], [0, 0, 0]], # instance 1
# 	[[6, 7, 8], [6, 5, 4]], # instance 2
# 	[[9, 0, 1], [3, 2, 1]], # instance 3
# 	])
# # sequence_batch = [2,1,2,2]
# sequence_batch = [1,2,2,2]
#
# with tf.Session() as sess:
# 	sess.run(init)
# 	output_eval1, state_eval1 = sess.run([output,state],feed_dict={X:X_batch,sequence_length:sequence_batch})
cell = tf.contrib.rnn.BasicLSTMCell(num_units = n_neuron,)
input1 = tf.Variable(tf.ones([batch_size,n_input]))
h = tf.Variable(tf.ones([batch_size,n_neuron]))
c = tf.Variable(tf.ones([batch_size,n_neuron]))
initial_state = cell.zero_state(batch_size, tf.float32)
initial_state1 = h,c
a = cell(input1,initial_state1)
init = tf.global_variables_initializer()
# with tf.Session() as sess:
sess= tf.Session()
sess.run(init)
x1 = sess.run(initial_state)
# output_eval1, state_eval1 = sess.run([output,state],feed_dict={X:X_batch,sequence_length:sequence_batch})
a_eval, b_eval= sess.run(a)




n_steps = 2
n_inputs = 3
n_neurons = 5
n_neuron = 5
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
init = tf.global_variables_initializer()

X_batch = np.array([
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])

sess =tf.Session()
sess.run(init)
outputs_val,states_val = sess.run([outputs, states],feed_dict={X: X_batch})


from keras.layers import LSTM, Input
from keras.models import Model
input_layer = Input(shape=[ n_steps, n_inputs])
cel  = LSTM(n_neuron,return_state = True,return_sequences= True)(input_layer)
model = Model(input_layer,cel)
a= model.predict(X_batch)


