import os
import sys
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fftpack import dct
import time
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

nb_classes = 16
df = pd.read_csv("split_data_102_fixed/12_102-1.csv", header=None, index_col=False)
sample = df.iloc[:,0].values
sample = sample / max(abs(sample))
mfccs = np.zeros([32, 4])
for j in range(0, 32):
    temp = sample[j * 24 : (j + 1) * 24]
    temp_dct = dct(temp)
    mfccs[j, :] = temp_dct[0:4]

features = mfccs[np.newaxis, :, :]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

test_X1 = features

learning_rate = 0.00025
training_iters = 50000
batch_size = 150
display_step = 100
n_input = 4
n_steps = 32
n_hidden = 100
n_classes = 16
dropout = 0.5
regularization = 0.25

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

weight_h = tf.Variable(tf.random_normal([n_input, n_hidden]))
bias_h = tf.Variable(tf.random_normal([n_hidden]))

def RNN(_x, weight, bias):

    _x = tf.transpose(_x, [1, 0, 2])
    _x = tf.reshape(_x, [-1, n_input])

    _x = tf.nn.relu(tf.matmul(_x, weight_h) + bias_h)
    _x = tf.split(_x, n_steps, 0)

    cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple = True)
    cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple = True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1, cell_2], state_is_tuple = True)
    output, state = tf.nn.static_rnn(cell, _x, dtype = tf.float32)
    last = output[-1]
    return (tf.matmul(last, weight) + bias)

prediction = RNN(x, weight, bias)

# Define loss and optimizer
tv = tf.trainable_variables()
loss_f = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
loss_m = regularization * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
loss_sum = loss_f + loss_m
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_sum)

# Evaluate model
class_name = app_list = ["Wunderlist","WinRAR","iTunes","notepad++","SumatraPDF","Dropbox","POWERPOINT","WINWORD","EXCEL","chrome","CCleaner64","vlc","keeper","WhatsApp","Steam","Skype"]
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
pred_argmax = tf.argmax(prediction, 1) + 1
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
confus_matrix = tf.confusion_matrix(tf.argmax(y,1), tf.argmax(prediction,1), 16)
# Initializing the variables
init = tf.global_variables_initializer()
class_dict = [1, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 9]

with tf.Session() as session:
    session.run(init)
    saver = tf.train.Saver()
    saver.restore(session,'model/model20211111-2-8000.ckpt') #调用模型
    pred_class = int(session.run(pred_argmax, feed_dict = {x : test_X1}))
    pred_class_real = class_dict[pred_class - 1]
    pred_class_name = class_name[pred_class - 1]
    print('Predicted class:', pred_class_real, pred_class_name)