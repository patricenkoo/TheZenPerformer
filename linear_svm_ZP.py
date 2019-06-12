from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape) 
    classes_num = W.shape[1]
    num_training = X.shape[0]
    loss = 0.0
    for i in range(num_training):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(classes_num):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 
            dW = (-1/classes_num) * margin
            if margin > 0:
                loss += margin
                dW += (-1/classes_num) * margin 
    loss /= num_training
    dW /= num_training
    
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape) 
    num_training = X.shape[0]
    scores = np.matmul(X, W)

    index = np.arange(num_training) 
    updated_score = scores[index,y] 

    transpose_Margin = np.maximum((scores.T - updated_score + 1),0) 
    margin = np.transpose(transpose_Margin) 
    margin[index,y] = 0 

    loss = np.sum(margin)/num_training
    loss += reg * np.sum(W * W)
    loss /= num_training

    dW /= num_training
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    binary = margin
    binary[margin>0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[index,y] = -row_sum
    dW = np.matmul(X.T, binary)

    dW /= num_training
    dW += reg * 2 * W

    return loss, dW
