import pyaudio
import wave
import os
import time
import sys
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fftpack import dct
import time
import tensorflow.compat.v1 as tf
import easygui
tf.disable_v2_behavior()
from tkinter import *


learning_rate = 0.00025
training_iters = 50000
batch_size = 150
display_step = 100
n_input = 4
n_steps = 32
n_hidden = 100
n_classes = 4
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
class_name = app_list = ["Wunderlist","POWERPOINT","WinRAR","Skype"]
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
pred_argmax = tf.argmax(prediction, 1) + 1
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# confus_matrix = tf.confusion_matrix(tf.argmax(y,1), tf.argmax(prediction,1), 16)
# Initializing the variables
init = tf.global_variables_initializer()
class_dict = [1, 14, 2, 9]



def start_audio(time = 15,save_file="temp.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 192000 #!!!!!!!!!!!!!!!!!!!!!!!!
    RECORD_SECONDS = time  #需要录制的时间
    WAVE_OUTPUT_FILENAME = save_file	#保存的文件名
    p = pyaudio.PyAudio()	#初始化
    print("ON")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)#创建录音文件
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)#开始录音
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("OFF")

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')	#保存
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



def callback_run():
    start_audio()
    os.system("matlab -nosplash -nodesktop -r combined")
    time.sleep(15)


    rootdir = "output_csv/"
    dirList = os.listdir(rootdir)
    for ii in range(len(dirList)):
        subpath = os.path.join(rootdir, dirList[ii])
        df = pd.read_csv(subpath, header=None, index_col=False)
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


        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            saver.restore(session,'model/model20220418-demo-5100-1.0.ckpt') #调用模型
            pred_class = int(session.run(pred_argmax, feed_dict = {x : test_X1}))
            pred_class_real = class_dict[pred_class - 1]
            pred_class_name = class_name[pred_class - 1]
            print('Predicted class:', pred_class_real, pred_class_name)
            # easygui.msgbox("打开程序:"+pred_class_name)
            os.remove(subpath)
            root1 = Tk(className = "PFC")
            w = 500
            h = 300
            root1.geometry("%dx%d" %(w, h))
            label = Label(root1, text="打开程序: "+pred_class_name, fg = 'black', font=("微软雅黑", 40,"bold"), wraplength=1000)
            label.place(x=50, y=50)
            root1.mainloop()
        # if os.path.exists(subpath):
        #     os.remove(subpath)

if __name__ == '__main__':
    root = Tk(className = "PFC")
    w = 1100
    h = 720
    root.geometry("%dx%d" %(w, h))
    label = Label(root, text=r"隔离网络信息监测", fg = 'black', font=("微软雅黑",70,"bold"), wraplength=1000)
    label.place(x=175, y=200)
    runButton = Button(root, text='运行',command=callback_run, height=2, width=10, font=("微软雅黑",30,"bold"))
    runButton.place(x=400, y=450)
    root.mainloop()

    # gui_judge = easygui.ccbox(msg="Application Inference with PFC Noise",title="PFC Inference",choices=["run","finish"])