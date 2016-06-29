# -*- coding: utf-8 -*-
"""
    preprocess2.py
    ==============

    The script to preprocess the User Identification From the Walking
    Activity Data Set, downloaded from 
    https://archive.ics.uci.edu/ml/datasets/User+Identification+From+Walking+Activity.

    Author: Chunxu Tang
    Email: chunxutang@gmail.com
    License: MIT
"""


import numpy as np
from numpy import mean, absolute


# Calculate the median absolute deviation of an array.
def mad(a):
    med = np.median(a)
    return np.median(np.absolute(a - med))


# Partition every user's data into sliding windows (100 samples) with
# a 50% overlap.
for j in range(1, 23):
    data = np.genfromtxt(str(j) + '.csv', delimiter=',')

    length = len(data)
    n_block = length / 50 + 1

    ret = np.zeros(shape=(n_block, 10))
    curr_index = 0
    i = 0
    while i < length:
        block = data[i:i+100, 1:4]
        val = np.array([j,
                        block[:, 0].mean(), block[:, 1].mean(), block[:, 2].mean(),
                        block[:, 0].std(), block[:, 1].std(), block[:, 2].std(),
                        mad(block[:, 0]), mad(block[:, 1]), mad(block[:, 2])])
        ret[curr_index] = val
        curr_index += 1
        i += 50
    
    np.savetxt(str(j) + '_ret.csv', ret, delimiter=',')


# Merge the 23 files into one large file.
with open("total.csv", "a") as file:
    for i in range(1, 23):
        with open(str(i)+'_ret.csv', 'r') as f:
            file.write(f.read())



