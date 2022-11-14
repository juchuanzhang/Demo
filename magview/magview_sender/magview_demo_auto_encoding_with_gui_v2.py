# -- coding: utf-8 --
from pandas import DataFrame
import os
import math
import numpy as np
import pandas as pd
import sys
import gradio as gr
import requests

fps = 60
time_len = 20
frame_total = 1200
# frame_total = 2202
def encode(s):
    data_without_fec = ''.join([bin(ord(c)).replace('0b', '').zfill(16) for c in s])
    # doing FEC
    r = 3
    n = int(math.pow(2, r)) - 1
    k = n - r 
    data_with_fec = np.zeros(math.ceil(len(data_without_fec) / k * n))
    hamming_cycles = len(data_without_fec) // k
    integer_power2 = np.ones(r, dtype="int")

    for i in range(1,r):
        integer_power2[i] = integer_power2[i - 1] * 2

    for i in range(0, hamming_cycles):
        xor_result = 0
        data_to_transfer_count = 0
        for j in range(0, n):
            if not ((j + 1) in integer_power2):
                # print(j)
                data_with_fec[i * n + j] = int(data_without_fec[i * k + data_to_transfer_count])
                data_to_transfer_count = data_to_transfer_count + 1
                if(data_with_fec[i * n + j] == 1):
                    xor_result = xor_result ^ (j+1)
            else:
                data_with_fec[i * n + j] = 0

        for j in range(0, r):
            if(xor_result % 2 == 1):
                data_with_fec[i * n + integer_power2[j] - 1] = 1
            xor_result = xor_result // 2

    return data_with_fec

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

# os.chdir(r"C:\Users\Administrator.DESKTOP-8L1OBSS\Desktop\magview_demo")
# os.chdir(r"")

def timestamp_generation(text):
    # text = sys.argv[1]
    print(text)
    message = encode(text)
    print(len(message))
    preamble = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    other = np.round(np.random.rand((time_len-10)*fps - len(message) - len(preamble)))
    d = pd.read_csv("message_transfer.csv", sep='.')
    data = d['true_number'].values
    print(data)
    file = open(r'timestamp_message_transfer.txt','w')
    file.write("# timestamp format v2")
    Interval = 1000/fps
    before = 5
    after = 5
    for i in np.arange(0, before * fps, 1):
         file.write("\n")
         file.write(str(i*Interval))

    for i in np.arange(0, (time_len - 10) * fps, 1):
        if(data[i] == 1):
            file.write("\n")
            file.write(str(i*Interval + 3 + before * fps * Interval))
        else:
            file.write("\n")
            file.write(str(i*Interval - 3 + before * fps * Interval))

    for i in np.arange(0 , (frame_total - 1200) + after * fps , 1):
        file.write("\n")
        file.write(str((i + (frame_total - after * fps ))*Interval))
    file.close()

    ss = decode(message)
    print( ss)
    os.system("mkvmerge --timestamps 0:timestamp_message_transfer.txt demo_20s.mp4 -o output.mkv")
    return "编码成功！"

if __name__ == '__main__':

    # text = "欢迎来到智能系统安全实验室"
    # text = gr.inputs.Text()
    gr.Interface(fn=timestamp_generation, inputs="text", outputs="text", layout="vertical", title="隔离网络信息泄露").launch()
