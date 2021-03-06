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
        self.layer = []
        self.delta_hidden_layer = []

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
        self.weight_input_prev = self.weight_input
        for i in range(len(self.hidden_layer)):
            if i > 0:
                weight_layer_temp = np.random.uniform(-.5,.5,[self.hidden_layer[i-1],self.hidden_layer[i]])
                self.weight_hidden_layer.append(weight_layer_temp)

        self.weight_hidden_layer_prev = self.weight_hidden_layer
        self.weight_output = np.random.uniform(-.5,.5,[self.hidden_layer[-1],1])
        self.weight_output_prev = self.weight_output
        for i in range(epoch):
            for j in range(len(self.input)):
                self.iterator = j
                self.batch_output = []
                self.batch_hidden_layer = []
                self.batch_error = []
                for k in range(batch_size):
                    self.feedforward()
                    self.backprop_search_gradient()
                print("Epoch-",epoch," :")
                print(np.average(self.batch_error))
                self.backprop_update_weight()

    def feedforward(self):
        for i in range(self.n_hidden_layer):
            if i == 0:
                self.layer.append(sigmoid(np.dot(self.input[self.iterator], self.weight_input)))
            else:
                self.layer.append(sigmoid(np.dot(self.layer[i-1], self.weight_hidden_layer[i-1])))

        output_layer = sigmoid(np.dot(self.layer[-1], self.weight_output))
        self.output = output_layer
        error = (self.y[self.iterator] - self.output) * (self.y[self.iterator] - self.output) / 2
        self.batch_error.append(error)

    def backprop_search_gradient(self):
        for i in range(self.n_hidden_layer+2):
            if (i == 0):
                self.delta_output = (self.y[self.iterator] - self.output) * sigmoid_derivative(self.output)
            elif (i == 1):
                self.delta_hidden_layer.append(np.dot(self.weight_output, self.delta_output) * sigmoid_derivative(self.output))
            elif (i == self.n_hidden_layer+1):
                self.delta_hidden_layer.append(np.dot(self.weight_input, self.delta_hidden_layer[i-2]) * sigmoid_derivative(self.output))
            else:
                self.delta_hidden_layer.append(np.dot(self.weight_hidden_layer[self.n_hidden_layer-i], self.delta_hidden_layer[i-2]) * sigmoid_derivative(self.output))

    def backprop_update_weight(self):
        for i in range(self.n_hidden_layer+1):
            if i == 0 :
                hidden_layer_unit = np.reshape(self.layer[self.n_hidden_layer-1], (len(self.layer[self.n_hidden_layer-1]),-1))
                delta_weight = (self.momentum * np.multiply(np.transpose(self.delta_output), hidden_layer_unit)) + (self.learning_rate * self.weight_output_prev)
                self.weight_output_prev = self.weight_output
                self.weight_output = self.weight_output + delta_weight
            elif i == self.n_hidden_layer :
                hidden_layer_unit = np.reshape(self.input[self.iterator], (len(self.input[self.iterator]),-1))
                delta_weight = (self.momentum * np.multiply(np.transpose(self.delta_hidden_layer[i-1]), hidden_layer_unit)) + (self.learning_rate * self.weight_input_prev)
                self.weight_input_prev = self.weight_input
                self.weight_input = self.weight_input + delta_weight
            else:
                hidden_layer_unit = np.reshape(self.layer[self.n_hidden_layer-i-1], (len(self.layer[self.n_hidden_layer-i-1]),-1))
                delta_weight = (self.momentum * np.multiply(np.transpose(self.delta_hidden_layer[i-1]), hidden_layer_unit)) +  + (self.learning_rate * self.weight_hidden_layer_prev[self.n_hidden_layer-i-1])
                self.weight_hidden_layer_prev[self.n_hidden_layer-i-1] = self.weight_hidden_layer[self.n_hidden_layer-i-1]
                self.weight_hidden_layer[self.n_hidden_layer-i-1] = self.weight_hidden_layer[self.n_hidden_layer-i-1] + delta_weight
        