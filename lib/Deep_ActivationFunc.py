# 概要: 活性化関数
# 作成: 2017/08/20
# 更新:
import numpy as np

#シグモイド関数
def sigmoid(x):
    ans = 1/(1+np.exp(-x))
    return ans

#恒等関数
def identity(x):
    return x
