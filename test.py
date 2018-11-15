from nn import NeuralNetwork

import numpy as np

# lst = ['a', 'b', 'c']

# pool = cycle(lst)

# for item in pool:
# 	print(item)

X = np.array([[0, 81, 75, 0]])
y = np.array([[1]])
nn = NeuralNetwork(X,y)
nn.add(7)

nn.fit(12,100,0.5,0.001)