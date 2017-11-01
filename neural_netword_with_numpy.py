import numpy as np

import math

import matplotlib.pyplot as plt
from sklearn.preprocessing  import OneHotEncoder


N = 100 # number of points per class
d0 = 2 # dimensionality
C = 3 # number of classes
X = np.zeros((d0, N*C)) # data matrix (each row = single example)

y = np.zeros(N*C, dtype='uint8') # class labels

for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
  y[ix] = j
X  = X.transpose() # 300x2
enc = OneHotEncoder(sparse=False)
y_true = enc.fit_transform(y.reshape(-1,1)) # 300x3


def softmax(x):
    x = np.exp(x)
    # x = np.exp(x -np.max(x, axis = 1, keepdims = True))
    x = x/np.sum(x,axis=1,keepdims=True)
    return x
# def relu(x):
#     return max(x,0)
# def logarit(x):
#     return np.lo

def loss1(W1,W2,b1,b2):
    Z1 = np.dot(X,W1) + b1 # 300 x 4
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(A1,W2) + b2 # 300 x 3
    A2 = softmax(Z2)

    loss = -np.multiply(np.log(A2),y_true)
    loss = np.sum(loss)/nb_example
    return loss
def check_gradient(x,W2,b1,b2):
    # x la W1
    # h = np.ones(shape=x.shape)*10e-6
    # gradient = (loss1(x+h,W2,b1,b2) -loss1(x-h,W2,b1,b2))/(2*h)
    shape_x = x.shape
    g = np.zeros(shape_x)
    for i in range(shape_x[0]):
        for j in range(shape_x[1]):
            x_neg = x.copy()
            x_pos = x.copy()
            x_neg[i,j] -= 10e-6
            x_pos[i,j] += 10e-6
            g[i,j]= (loss1(x_pos,W2,b1,b2) - loss1(x_neg,W2,b1,b2))/(2*10e-6)

    return g

def softmax_tiep(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

nb_example = X.shape[0]
d_hidden = 100
W1 = np.random.randn(2, d_hidden)
b1 = np.zeros(shape=(1,d_hidden))
W2 = np.random.randn(d_hidden, 3)
b2 = np.zeros(shape=(1,3))
learning_rate = 0.1

for i in range(100000):
    Z1 = np.dot(X,W1) + b1 # 300 x 4
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(A1,W2) + b2 # 300 x 3
    A2 = softmax(Z2)
    # A2_tiep = softmax_tiep(Z2.T)
    if i %10000 ==0:
        # loss = -np.multiply(np.log(A2),y_true)
        loss = -np.log(A2)*y_true
        loss = np.sum(loss)/nb_example
        x = loss1(W1,W2,b1,b2)
        print('tai step thu : {}, ta co loss : {}'.format(i,loss))

    # dZ2 = np.sum(A2 - y_true,axis=0) # 1 x3
    dZ2 = (A2 - y_true)/nb_example # 300 x3 (đoạn này giữ nguyện 300 x3 là vì coi như với giá trị của ma trận Z2 là ẩn, thì đạo hàm của 1 hàm scalar theo 1 ma trận là 1 ma trận có đọ dài bằng với ma trân input )
    dW2 = np.dot(A1.T,dZ2) # 4x3
    # db2 = np.sum(A2 - y_true,axis=0).reshape(3) # 1 x3
    db2 = np.sum(A2 - y_true,axis=0) # 1 x3

    dA1 = np.dot(dZ2,W2.T) # 300x4
    dZ1 = np.multiply(dA1,Z1>0)
    dW1 = np.dot(X.T,dZ1) # 2x4
    dW1_test = check_gradient(W1,W2,b1,b2)
    a = np.linalg.norm(dW1_test - dW1)
    # db1 = np.sum(dZ1,axis=0).reshape(4) #1 x4
    db1 = np.sum(dZ1,axis=0) #1 x4

    W1 = W1 - learning_rate*dW1
    W2 = W2 - learning_rate*dW2
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2


import numpy as np
rand_seq_len = np.random.choice(range(3,7))
seqlens.append(rand_seq_len)
rand_odd_ints = np.random.choice(range(1,10,2),
rand_seq_len)
rand_even_ints = np.random.choice(range(2,10,2),
rand_seq_len)