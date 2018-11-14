import numpy as np


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        # self.weights1 = np.random.uniform(-.5,.5,[self.input.shape[1],4])
        # self.weights1   = np.random.rand(self.input.shape[1],4)
        # self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.n_hidden_layer = 0
        self.hidden_layer = []
        self.weight_hidden_layer = []
        # self.weight_hidden_layer.append(self.weights1)
        # self.weight_hidden_layer.append(self.weights2)



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



    def feedforward(self):
        self.layer = []
        for i in range(self.n_hidden_layer):
            if i == 0:
                self.layer.append(sigmoid(np.dot(self.input, self.weights_input)))
            else:
                self.layer.append(sigmoid(np.dot(self.layer[i-1], self.weight_hidden_layer[i-1])))



        output_layer = sigmoid(np.dot(self.layer[-1], self.weight_output))
        # self.layer.append(output_layer)
        # self.output = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = output_layer

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
