from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.layers import  fully_connected
from datetime import datetime
import numpy as np

learning_rate = 0.1
n_neuron = 100
n_output = 10
batch_size = 125
n_step = 28
n_input = 28
n_epoch = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True )
X = tf.placeholder(tf.float32,[None,n_step,n_input], name='X')
y = tf.placeholder(tf.float32,[None,n_output], name='y')


basic_cell = tf.contrib.rnn.BasicRNNCell(n_neuron,)
output_rnn, state = tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)
output = tf.layers.dense(state, n_output,)
#
# output = tf.contrib.layers.fully_connected(
#     state,
#     n_output,
#     None
# )

# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#     labels=y,
#     logits=output,
#     name='loss'
# )
loss_raw = tf.nn.softmax_cross_entropy_with_logits(
    labels=y,
    logits=output,
    name='loss'
)
loss = tf.reduce_mean(loss_raw)

with tf.name_scope('accuracy'):
    # accuracy_raw = tf.nn.in_top_k(
    #     predictions=output,
    #     targets=y,
    #     k =1,
    # )
    accuracy_raw = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy_raw,tf.float32))
    # accuracy_raw =  tf.equal(tf.argmax(y_logit,1), tf.argmax(y,1))
    # accuracy = tf.reduce_mean(tf.cast(accuracy_raw,tf.float32), name='accuracy',)

with tf.name_scope('optimiser'):
    # optimiser = tf.train.AdamOptimizer()
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    op_train = optimiser.minimize(loss)

init = tf.global_variables_initializer()
# with tf.Session() as sess:
sess =  tf.Session()
sess.run(init)
for i in range(n_epoch):
    nb_sample = mnist.train.num_examples
    internal_step = nb_sample//batch_size
    for step in range(internal_step):
        X_batch,y_batch = mnist.train.next_batch(batch_size)
        X_batch = np.reshape(X_batch,[-1,28,28])
        sess.run(op_train, feed_dict={X:X_batch, y:y_batch})
        global_step = step + i*internal_step
        if global_step%1000 ==0:
            X_test=  mnist.test.images
            X_test=  np.reshape(X_test,[-1,28,28])
            y_test = mnist.test.labels
            sess.run(op_train, feed_dict={X: X_test, y: y_test})
            loss_eval ,accuracy_eval = sess.run([loss,accuracy],feed_dict={X:X_test, y:y_test})
            print('loss: {} va accuaray:{}  tai step: {}'.format(loss_eval ,accuracy_eval,global_step))

sess.close()