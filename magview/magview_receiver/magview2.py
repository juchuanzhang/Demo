import os
import math
import numpy as np
import pandas as pd
import csv
import HelperFunctions
import time
import sys
from pandas import Series,DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit,cross_val_score
import pandas as pd
import numpy as np
from tkinter import * 

home = r'C:\Users\usslab\Desktop\magview_demo_new'
fps = 60
fs = 10000
time_len = 20
res = ''
root = Tk(className = "magview")

def list2num(l):
    return int(l[0])
def decode(data_received):
    r = 3
    n = int(math.pow(2, r)) - 1
    k = n - r
    data_decoded = ""
    hamming_cycles = len(data_received) // n
    # print("hamming_cycles=", hamming_cycles)
    integer_power2 = np.ones(r, dtype="int")

    for i in range(1, r):
        integer_power2[i] = integer_power2[i - 1] * 2

    for i in range(0, hamming_cycles):
        xor_result = 0
        xor_extract = 0
        data_to_transfer_count = 0
        for j in range(0, n):
            if not ((j + 1) in integer_power2):
                if (data_received[i * n + j] == 1):
                    xor_result = xor_result ^ (j + 1)
            else:
                # print("i=", i)
                # print(i * n + j)
                xor_extract = xor_extract + int((j + 1) * data_received[i * n + j])
        # print(type(xor_result), type(xor_extract))
        xor_decoded = xor_result ^ xor_extract
        if (xor_decoded == 0):
            # print("no error!")
            pass
        else:
            # print("No."+str(xor_decoded)+"is error!")
            data_received[i * n + xor_decoded - 1] = int(data_received[i * n + xor_decoded - 1]) ^ 1
            # pass
        data_decoded_count = 0
        for j in range(0, n):
            if not ((j + 1) in integer_power2):

                data_decoded = data_decoded + (str(int(data_received[i * n + j])))
                data_decoded_count = data_decoded_count + 1

    return ''.join([chr(i) for i in [int(b, 2) for b in [data_decoded[16*k:16*(k+1)] for k in range(0, math.floor(len(data_decoded) / 16))]]])



#将真实的编码位读取进来
def getTrueData():
    os.chdir(home)
    file = pd.read_csv(r"message_transfer.csv")
    df = pd.DataFrame(file)
    truedata = df.values.tolist()
    truedata = list(map(list2num, truedata))
    truedata = truedata[0:600]
    preamble = truedata[0:50]
    return truedata, preamble

#将采集到的bin文件以int16位的格式读入,并过滤掉头信息等
def getSampledData(filename):
    data = []
    with open(filename, 'rb') as f:
        while True:
            tmp = f.read(2)
            if not tmp: break
            tmp = int.from_bytes(tmp, 'little')
            #tmp = ((tmp << 8) | (tmp >> 8)) & 0xFFFF
            data.append(tmp)
    data_skip = 1690 * 8 + 4
    data = data[data_skip : -1]
    data = data[0 : time_len * fs]
    def f(a):
        return a / (4096 * 10)
    data = list(map(f, data))
    return data

#通过互相关，找到开始编码的位置
def findStartPos(data, preamble):
    N = len(data)
    data_preamble = data[0 : fs * 10]
    oneBitLen = 10000/fps
    sampleInterval = oneBitLen / 10
    window_length = round(50 * (oneBitLen))
    window_overlap = 10
    window_end = len(data_preamble) - window_length
    corr_results = []
    preamble = pd.Series(preamble)
    for ii in range(math.floor(window_end / window_overlap)):
        window_data_origin = data_preamble[ii*window_overlap : ii*window_overlap+window_length]
        window_data_sample = []
        for k in range(50*10):
            window_data_sample.append(window_data_origin[math.floor(sampleInterval * k)])
        sum_data = []
        for t in range(50):
            sumOfOneBit = 0
            for r in range(10):
                sumOfOneBit = sumOfOneBit + window_data_sample[t * 10 + r]
            sum_data.append(sumOfOneBit)
        sum_data = pd.Series(sum_data)
        corr_results.append(round(preamble.corr(sum_data), 4))
    corr_results = list(map(math.fabs, corr_results))
    return corr_results, window_overlap


#将采集到的数据按位分片
def splitData(corr_results, truedata, window_overlap, data):
    offset = (corr_results.index(max(corr_results)) * window_overlap - 50000) / 10000 - 0.0012
    outdir = r'split_x395_60p_small'
    print(len(truedata))
    for i in range(len(truedata)):
        start = 1 / fps * (i + 5 * fps) + offset - 0.001
        start_index = round(start * fs + 1)
        split_data = data[start_index : start_index + math.floor(1 / fps * fs)]
        filename = str(i) + '.csv'
        df = pd.DataFrame(split_data)
        df.to_csv(os.path.join(outdir, filename), index=False, header=False)

#提取特征
def extract_features():
    feature_num = 10
    sample_len = 167
    # os.chdir(r'C:\Users\Administrator.DESKTOP-8L1OBSS\Desktop\magview_demo')
    i = 0
    ground_truth = pd.read_csv(r'message_transfer.csv')
    alldata = pd.DataFrame(columns=['label'] + [x for x in range(0, feature_num)])
    for i in range(0, 600):
        indir = 'split_x395_60p_small\\'
        data_read = pd.read_csv(indir + str(i) + '.csv', header=None)
        data_read.rename(columns={0: 'amplitude'}, inplace=True)
        results = HelperFunctions.extract_features(data_read, feature_num)
        results['label'] = int(ground_truth['true_number'][i])
        alldata = alldata.append(results, ignore_index=True)
        # if i % 100 == 0:
        #     print('File %d is completed!' %(i))
    alldata.to_csv('features\\split_x395_60p_small_feature_' + str(feature_num) + '.csv', index=False)

#通过svm训练分类
def train():
    training_set_num = 50
    indir = r'features\split_x395_60p_small_feature_10.csv'
    data = pd.read_csv(indir)
    data_train = data.iloc[0:training_set_num][:]
    data_test = data.iloc[training_set_num:len(data)][:]

    dft=data_test.values
    Xt=dft[:,1:]  #取出测试集中所有的信号值
    yt=dft[:,0]  #取出真实值

    from sklearn import preprocessing
    scaler=preprocessing.StandardScaler().fit(Xt)  #计算Xt的均值和方差
    Xt=scaler.transform(Xt)  #根据计算出来的均值和方差对数据进行转换

    #part4-----train
    clf=SVC(probability=True)
    df=data_train.values
    X=df[:,1:]
    y=df[:,0]
    X=scaler.transform(X)
    clf.fit(X,y)
    from sklearn.metrics import accuracy_score
    predict_results = np.c_[yt, clf.predict(Xt)]
    #np.savetxt('predict_results.csv', predict_results, fmt='%d', delimiter=',')
    score = accuracy_score(yt, clf.predict(Xt))
    print('training accuracy: ', accuracy_score(y, clf.predict(X)))
    print('validating accuracy: ', score)
    predict_results = np.transpose(predict_results)
    #print(predict_results[1][:713])
    return decode(predict_results[1][0:364])
def main():
    truedata, preamble = getTrueData()
    data = getSampledData("1.bin")
    corr_results, window_overlap = findStartPos(data, preamble)
    splitData(corr_results, truedata, window_overlap, data)
    extract_features()
    res = train()
    label = Label(root, text=res, fg = 'red', font=( "微软雅黑", 48, "bold"), wraplength=1000)
    label.place(x=175, y=200)

if __name__ == "__main__":
    # root = Tk(className = "magview")
    w = 1280
    h = 720
    root.geometry("%dx%d" %(w,h))
    start = Button(root, width = 15, height = 10, text = 'start', anchor = 'c', padx = 5, pady = 5, fg = 'red', command = main) 
    start.place(x=10, y=10)                    
    root.mainloop()            


   