import numpy as np
from collections import deque


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y, learning_rate, momentum):
        self.input      = x
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_hidden_layer = 0
        self.hidden_layer = []
        self.weight_hidden_layer = []



    def add(self,unit):
        if self.n_hidden_layer < 10:
            self.hidden_layer.append(unit)
            self.n_hidden_layer +=1

    def fit(self, batch_size, epoch):
        self.batch_size = batch_size
        self.epoch = epoch
        print ("self.hidden_layer[0]", self.hidden_layer[0])
        self.weights_input = np.random.uniform(-.5,.5,[self.input.shape[1],self.hidden_layer[0]])
        # self.weight_hidden_layer.append(self.weights_input)
        for i in range(len(self.hidden_layer)):
            if i > 0:
                weight_layer_temp = np.random.uniform(-.5,.5,[self.hidden_layer[i-1],self.hidden_layer[i]])
                self.weight_hidden_layer.append(weight_layer_temp)

        self.weight_output = np.random.uniform(-.5,.5,[self.hidden_layer[-1],1])
        # self.weight_hidden_layer.append(self.weight_output)

        print ("self.weight_hidden_layer\n", self.weight_hidden_layer)

        self.feedforward()
        self.backprop()


    def feedforward(self):
        self.hidden_layer_out = []
        self.hidden_layer_in = []
        for i in range(self.n_hidden_layer):
            if i == 0:
                self.hidden_layer_in.append(np.dot(self.input, self.weights_input))
                self.hidden_layer_out.append(sigmoid(self.hidden_layer_in[i]))
            else:
                self.hidden_layer_in.append(np.dot(self.hidden_layer_out[i-1], self.weight_hidden_layer[i-1]))
                self.hidden_layer_out.append(sigmoid(self.hidden_layer_in[i]))

        self.output_layer_in = np.dot(self.hidden_layer_out[-1], self.weight_output)
        self.output_layer_out = sigmoid(self.output_layer_in)
        self.output = self.output_layer_out



    def backprop(self):
        deq = deque([])
        # deq.appendleft(1)

        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        self.o1 = (self.y - self.output) * sigmoid_derivative(self.output_layer_in)
        # self.o1 = (self.y - self.output) * sigmoid_derivative(self.output)
        self_hidden_unit = deque([])
        for i in range(self.n_hidden_layer-1,-1,-1):
            self_hidden_unit.appendleft(sigmoid_derivative(self.output_layer_in[i]) * )
        # for i in range(self.n_hidden_layer):


        #
        # d_weights2 = np.dot(self.hidden_layer_out1.T, ((self.y - self.output) * sigmoid_derivative(self.output)))
        # d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.hidden_layer_out1)))
        # # weight_output = np.dot( ,(self.o1 = (self.y - self.output) * sigmoid_derivative(self.output)))
        #
        # # if (batch_size==1): #stochastic
        # #
        # # else:
        #
        # # update the weights with the derivative (slope) of the loss function
        # self.weights1 += d_weights1
        # self.weights2 += d_weights2
