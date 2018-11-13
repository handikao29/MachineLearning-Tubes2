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
