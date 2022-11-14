# import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
import pandas as pd

train_dir = "train"

test_dir1 = "test"
gpu_options = tf.GPUOptions(allow_growth=True)
#test_dir2 = "high_test"
#test_dir3 = "low_snr_test_sofa"
#test_dir4 = "low_snr_test_backroom"
#tf.disable_eager_execution()
test_X1 = np.load("features/"+ test_dir1 + "_features.npy")
print("Test Features: ",test_X1.shape)
test_Y1 = np.load("features/" + test_dir1 + "_labels.npy")
print("Test Labels: ",test_Y1.shape)
#test_X2 = np.load("renamed_no6/"+ test_dir2 + "_features.npy")
#print("Test Features: ",test_X2.shape)
#test_Y2 = np.load("renamed_no6/" + test_dir2 + "_labels.npy")
#print("Test Labels: ",test_Y2.shape)
#test_X3 = np.load("features1/"+ test_dir3 + "_features.npy")
#print("Test Features: ",test_X3.shape)
#test_Y3 = np.load("features1/" + test_dir3 + "_labels.npy")
#print("Test Labels: ",test_Y3.shape)
#test_X4 = np.load("features1/"+ test_dir4 + "_features.npy")
#print("Test Features: ",test_X4.shape)
#test_Y4 = np.load("features1/" + test_dir4 + "_labels.npy")
#print("Test Labels: ",test_Y4.shape)

train_X = np.load("features/" + train_dir + "_features.npy")
print("Train Features: ",train_X.shape)
train_Y = np.load("features/" + train_dir + "_labels.npy")
print("Train Labels: ",train_Y.shape)
#sys.exit()

#learning_rate = 0.00025
learning_rate = 0.00025
training_iters = 50000
batch_size = 150
display_step = 100
n_input = 4
n_steps = 32
n_hidden = 100
n_classes = 16
#dropout = 0.5
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
    #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = dropout)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1, cell_2], state_is_tuple = True)
    output, state = tf.nn.static_rnn(cell, _x, dtype = tf.float32)

    #output = tf.transpose(output, [1, 0, 2])
    #last = tf.gather(output, (int)(output.get_shape()[0]) - 1)
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
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
pred_y = y;
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
confus_matrix = tf.confusion_matrix(tf.argmax(y,1), tf.argmax(prediction,1), 16)
# Initializing the variables
init = tf.global_variables_initializer()
#results = pd.DataFrame(columns=["Iter", "Training Accuracy",
#    "Testing Accuracy 1", "Testing Accuracy 2", "Testing Accuracy 3", 
#    "Testing Accuracy 4", "Loss", "RLoss"])
# columnsa = [str(x) for x in range(1, 17)]
# columnsa.insert(0, "Iter");
# pred_df = pd.DataFrame(columns=["Iter", str(x) for x in range(1, 17)])

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    session.run(init)
    saver = tf.train.Saver()
    saver.restore(session,'model/model_0317_9600_0.86249995.ckpt') #调用模型
    """
    for itr in range(training_iters):
        offset = (itr * batch_size) % (train_Y.shape[0] - batch_size)
        batch_x = train_X[offset:(offset + batch_size), :, :]
        batch_y = train_Y[offset:(offset + batch_size), :]
        _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})

        if itr % display_step == 0:
            # Calculate batch accuracy
            acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            # loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            loss_ff = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            loss_mm = session.run(loss_m, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(itr) + ", Minibatch Loss= " + \
                  "{}".format(loss_ff) + ", Regularization Loss= " + \
                  "{}".format(loss_mm) + ", Training Accuracy= " + \
                  "{}".format(acc))
        if itr % 1000 == 0:
            #saver.save(session,'model_rnn')
            tacc1 = session.run(accuracy, feed_dict={x: test_X1, y: test_Y1})
            tacc2 = session.run(accuracy, feed_dict={x: test_X2, y: test_Y2})
            # tacc3 = session.run(accuracy, feed_dict={x: test_X3, y: test_Y3})
            # tacc4 = session.run(accuracy, feed_dict={x: test_X4, y: test_Y4})
            acc = session.run(accuracy, feed_dict={x: train_X, y: train_Y})
            loss_ff = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            loss_mm = session.run(loss_m, feed_dict={x: batch_x, y: batch_y})
            cm = session.run(confus_matrix, feed_dict={x: test_X1, y: test_Y1})
            pred = session.run(prediction, feed_dict={x: test_X1})
            pred_y1 = session.run(pred_y, feed_dict={x: test_X1, y: test_Y1})
            print('validate accuracy: ', tacc1)
            print('test accuracy: ', tacc2)
            #print('Test accuracy sofa: ', tacc3)
            #print('Test accuracy backroom: ', tacc4)
            #print('Train accuracy: ', acc)
            print('confusion matrix: ', cm)
            #print('prediction: ', pred)
            # print('predarg', pred_arg1)
            # print('y', pred_y1)
            
            results = results.append({"Iter":itr, "Training Accuracy": acc, "Testing Accuracy 1": tacc1, 
               "Loss": loss_ff, "RLOSS": loss_mm},ignore_index=True)
            np.savetxt('results1/Confusion_matrix_'+str(itr)+'.csv', cm, fmt = '%3d',  delimiter = ',')
            np.savetxt('results1/Prediction_'+str(itr)+'.csv', pred, delimiter = ',')
            np.savetxt('results1/y_'+str(itr)+'.csv', pred_y1, fmt = '%3d', delimiter = ',')        
            # print('Test accuracy: ',session.run(accuracy, feed_dict={x: test_X, y: test_Y}))
    """
    #print('validate accuracy: ',session.run(accuracy, feed_dict={x: test_X1, y: test_Y1}))
    print('test accuracy: ',session.run(accuracy, feed_dict={x: test_X1, y: test_Y1}))
    # print('Test accuracy sofa: ',session.run(accuracy, feed_dict={x: test_X3, y: test_Y3}))
    # print('Test accuracy backroom: ',session.run(accuracy, feed_dict={x: test_X4, y: test_Y4}))
    #print('Train accuracy: ',session.run(accuracy, feed_dict={x: train_X, y: train_Y}))
    #results.to_csv("results1/20210313.csv")      
    #saver = tf.train.Saver()
    #saver.save(session,'final_model_rnn.ckpt')#保存模型


