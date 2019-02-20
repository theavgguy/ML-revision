import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#define shapes
N = 200
M = 4
itr = 10000
lr = 0.0005

#read data
df = pd.read_csv("Advertising.csv")
mat = df.values

X = np.ones((N,M))
targets = np.ones((N,1))
for i in range(N):
    targets[i] = mat[i][4]
    for j in range(1,4):
        X[i][j-1] = mat[i][j]

#normalize data
def normalize():
    mean = np.mean(X,0)
    mx = np.max(X,0)
    mn = np.min(X,0)
    mean[3] = 0
    mn[3] = 0
    for i in range(N):
        for j in range(M):
            X[i][j] = (X[i][j]-mean[j])/(mx[j]-mn[j])

#variables
W = np.zeros((M,1))

#forward prop
def forward_prop(X,W):
    predicted = np.dot(X,W)
    return predicted

#loss function 
def MSE_loss(t,p):
    loss = 0
    for i in range(N):
        loss = loss+((t[i]-p[i])**2)
    loss = loss/(2*N)
    return loss

#gradients with respect to each weight and bias
def gradients(t,p,X,lr):
    error = p-t
    gradients = np.dot(error.T,X)
    gradients = gradients*(lr/N)
    return gradients

#gradient descent update rule
def update_rule(W,G):
    W = W-G.T
    return W

#train the model
loss = []
iterations = []
normalize()
for i in range(itr):
    predictions = forward_prop(X,W)
    gradient = gradients(targets,predictions,X,lr)
    W = update_rule(W,gradient)
    loss.append(MSE_loss(targets,predictions))
    iterations.append(i)
    if(i%100)==0:
        print("Iteration = ",i)
        print("Error = ",loss[-1])

plt.plot(iterations,loss)