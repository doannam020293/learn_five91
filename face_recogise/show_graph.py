import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename =r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\sofware\software\real time face recognition\20170511-185253\20170511-185253.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR=r'C:\nam\work\tf_log'
train_writer = tf.summary.FileWriter(LOGDIR,tf.get_default_graph())
# train_writer.add_graph(sess.graph,)

file_writer.close()

# model_exp = os.path.expanduser(model)
# if (os.path.isfile(model_exp)):
#     print('Model filename: %s' % model_exp)
#     with gfile.FastGFile(model_exp, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')
