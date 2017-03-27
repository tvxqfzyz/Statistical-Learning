#necessary library
import numpy as np
from chainer import datasets

#load daset
train, test = datasets.get_mnist()

#construct farir list of array for both training and testing
x_train = []
y_train = []
x_test = []
y_test = []
for x,y in train:
        x_train.append(x)
        y_train.append(y)
for x,y in test:
        x_test.append(x)
        y_test.append(y)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

#record size of the model
training_size = x_train.shape[0] #60,000
input_dimension = x_train.shape[1] #784 = 28*28 = d 
output_dimension = np.unique(y_train).shape[0] #10 = 0~9 = K
#learning rate
l_r = 0.15
#number of layers (could be changed as request)
layer = 200

def activate(z):
        #sigmoid function
    #return 1.0/(1.0+np.exp(-z))
    #RELU
    return np.maximum(z,0)

def prime_activate(z):
    #sigmoid
    #return activate(z)*(1-activate(z))
    #RELU
    return z>0

def softmax(z):
        a = np.exp(z)
        a = a/a.sum()
        return a

def forward_pass(x,w1,b1,w2,b2):
        z= np.dot(w1,x)+b1
        a = activate(z)
        a_prime = prime_activate(z)
        z_2 = np.dot(w2,a)+b2
        g = softmax(z_2)
        return g,a,a_prime

def back_propagation(x,y,w2,g,a,a_prime):
        delta_1 = np.zeros(output_dimension)
        delta_2 = np.zeros(layer)
        for i in xrange(output_dimension):
                delta_1[i] = 1.0*(y == i)-g[i]
        for j in xrange(layer):
                delta_2[j] = a_prime[j]*np.dot(w2.transpose()[j],delta_1)
        #calculate the gradient
        w1_grad = np.dot(x[:,None],delta_2[None,:]).transpose()
        w2_grad = np.dot(a[:,None],delta_1[None,:]).transpose()
        b1_grad = delta_2
        b2_grad = delta_1
        return w1_grad, w2_grad,b1_grad,b2_grad

def training(w1,w2,b1,b2):
    #pick up the orders stochastically
        stochastic_index = np.random.permutation(training_size)
        for index in stochastic_index:
                x = x_train[index]
                y = y_train[index]
                g,a,a_prime = forward_pass(x,w1,b1,w2,b2)
                w1_grad, w2_grad,b1_grad,b2_grad = back_propagation(x,y,w2,g,a,a_prime)
                #update the weight and the bias with gradient
                w1 += l_r*w1_grad
                w2 += l_r*w2_grad
                b1 += l_r*b1_grad
                b2 += l_r*b2_grad
        return w1,w2,b1,b2
def testing(w1,w2,b1,b2):
        correct_test = 0.0
        for i in xrange(y_test.shape[0]):
                x = x_test[i]
                y = y_test[i]
                g = forward_pass(x,w1,b1,w2,b2)[0]
                y_hat = g.argmax()
                correct_test += 1.0*(y==y_hat)
                accurancy = correct_test/(1.0*y_test.shape[0])
        return accurancy
#Step1: initialize weights and bias using random in numpy
w1 = np.divide(np.random.rand(layer,input_dimension),input_dimension)# initially the smaller weight is better
w2 = np.random.rand(output_dimension,layer)
b1 = np.random.rand(layer)
b2 = np.random.rand(output_dimension)
#Step 2: Training once and test the accurancy
for epoch in xrange(10):
        w1,w2,b1,b2 = training(w1,w2,b1,b2)
        test_accurancy = testing(w1,w2,b1,b2)
        print epoch, "Test accurancy: "+str(test_accurancy)








