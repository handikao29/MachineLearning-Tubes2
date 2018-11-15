import numpy as np
import math


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.n_hidden_layer = 0
        self.hidden_layer = []
        self.weight_hidden_layer = []



    def add(self,unit):
        if self.n_hidden_layer < 10:
            self.hidden_layer.append(unit)
            self.n_hidden_layer +=1

    def fit(self, batch_size, epoch, momentum, learning_rate):
        self.batch_size = batch_size
        self.epoch = epoch
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_input = np.random.uniform(-.5,.5,[self.input.shape[1],self.hidden_layer[0]])
        for i in range(len(self.hidden_layer)):
            if i > 0:
                weight_layer_temp = np.random.uniform(-.5,.5,[self.hidden_layer[i-1],self.hidden_layer[i]])
                self.weight_hidden_layer.append(weight_layer_temp)

        self.weight_output = np.random.uniform(-.5,.5,[self.hidden_layer[-1],1])

        for i in range(epoch):
            self.feedforward()
            self.backprop()

    def feedforward(self):
        self.layer = []
        for i in range(self.n_hidden_layer):
            if i == 0:
                self.layer.append(sigmoid(np.dot(self.input, self.weight_input)))
            else:
                self.layer.append(sigmoid(np.dot(self.layer[i-1], self.weight_hidden_layer[i-1])))

        output_layer = sigmoid(np.dot(self.layer[-1], self.weight_output))
        self.output = output_layer
        print(self.output)

    def backprop(self):

        self.delta_hidden_layer = []
        for i in range(self.n_hidden_layer+1):
            if (i == 0):
                self.delta_output = (self.y - self.output) * sigmoid_derivative(self.output)
            elif (i == 1):
                self.delta_hidden_layer.append(np.dot(self.weight_output, self.delta_output) * sigmoid_derivative(self.output))
            else:
                self.delta_hidden_layer.append(np.dot(self.weight_hidden_layer[self.n_hidden_layer-i], self.delta_hidden_layer[i-2]) * sigmoid_derivative(self.output))

        for i in range(self.n_hidden_layer):
            if i == 0 :
                self.weight_output = self.weight_output + (self.momentum * np.multiply(np.transpose(self.delta_output), np.transpose(self.layer[self.n_hidden_layer-1])))
            elif i == self.n_hidden_layer-1 :
                self.weight_input = self.weight_input + (self.momentum * np.multiply(np.transpose(self.delta_hidden_layer[i]), np.transpose(self.layer[0])))
            else:
                self.weight_hidden_layer[self.n_hidden_layer-i-1] = self.weight_hidden_layer[self.n_hidden_layer-i-1] + (self.momentum * np.multiply(np.transpose(self.delta_hidden_layer[i-1]), np.transpose(self.layer[self.n_hidden_layer-i-1])))


        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        # d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        # d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # # update the weights with the derivative (slope) of the loss function
        # self.weights1 += d_weights1
        # self.weights2 += d_weights2