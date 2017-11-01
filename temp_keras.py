# from keras.layers import Dense, Dropout, Input, LSTM
# from keras.models import Model
# from keras.layers import TimeDistributed
# import keras
#
# inputs = Input(shape=(784,))
#
# LSTM()
# x = Dense(units=256,activation='relu')(inputs)
# x = Dropout(0.5)(x)
# predict = Dense(units=1,activation='sigmoid')(x)
#
# model = Model(inputs=inputs, outputs=predict)
# model.inputs
#
# x = Input(shape=(784,))
# y = model(x)
#
#
# input_sequence = Input((20,784))
# seq = TimeDistributed(model)(input_sequence)
import os
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile


import keras

# a.save()


import sys
import os

PACKAGE_PARENT='..'
SCRIPT_DIR=os.path.dirname(os.path.realpath(os.path.join(os.getcwd(),os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR,PACKAGE_PARENT)))
sys.path.append(r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\real_time_deep_face_recognition_master')

from real_time_deep_face_recognition_master import facenet

from

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:

            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

        model = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\sofware\software\real time face recognition\20170511-185253\20170511-185253.pb'
        facenet.load_model(modeldir)


