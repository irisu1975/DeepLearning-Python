# 概要: 活性化関数
# 作成: 2017/08/20
# 更新: 2017/08/27
import numpy as np

#シグモイド関数
def sigmoid(x):
    ans = 1/(1+np.exp(-x))
    return ans

#恒等関数
def identity(x):
    return x

#ソフトマックス関数
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
