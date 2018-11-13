import random

class NN:

    def __init__(self, data, label, epoch, momentum, batch_size):
        self.data = data
        self.label = label
        self.epoch = epoch
        self.momentum = momentum
        self.batch_size = batch_size
        self.n_hidden_layer = 0
        self.hidden_layer = []

    def add(self,unit):
        if self.n_hidden_layer < 10:
            self.hidden_layer.append(unit)
            self.n_hidden_layer +=1

    def fit(self, x, y):
        self.array_of_node = []
        self.num_node = 0
        input_node = len(x[0])
        input_array = []
        for i in range(input_node):
            input_array.append(i)
            self.num_node += 1
        self.node_per_layer = []
        # self.index_per_layer =
        self.index_per_layer = [[]]
        self.index_per_layer.append(input_array)
        self.index_per_layer.remove([])

        # add hidden layer
        for n_node in self.hidden_layer:
            # print ("n_node", n_node)
            temp_array = []
            temp_val = self.num_node
            for i in range(n_node):
                temp_array.append(i+self.num_node)
                # self.num_node += 1
                temp_val += 1
            self.num_node = temp_val
            self.index_per_layer.append(temp_array)

        # add output layer
        output_array = [self.num_node]
        self.index_per_layer.append(output_array)

        # create adjacent matrix

        self.adjacent_matrix = [[0] * self.num_node] * self.num_node
        for row in range(self.num_node):
            for column in range(row+1):
                self.adjacent_matrix[row][column] = random.uniform(-0.05, 0.05)


        print (self.adjacent_matrix)
