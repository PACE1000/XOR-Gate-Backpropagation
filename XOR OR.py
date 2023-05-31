"""Kelompok Tugas Komputasi Cerdas

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Data XOR
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

# Menginisialisasi bobot dan bias untuk neuron dalam jaringan. Bobot dan bias diinisialisasi secara acak.
inputLayer_neurons = 2 #jumlah neuron di input layer
hiddenLayer_neurons = 2 #jumlah neuron di hidden layer
outputLayer_neurons = 1 #jumlah neuron di output layer

weights_input_hidden = np.random.uniform(size=(inputLayer_neurons,hiddenLayer_neurons))
weights_hidden_output = np.random.uniform(size=(hiddenLayer_neurons,outputLayer_neurons))

bias_hidden =np.random.uniform(size=(1,hiddenLayer_neurons))
bias_output = np.random.uniform(size=(1,outputLayer_neurons))

print(weights_input_hidden)
print(weights_hidden_output)
print(bias_hidden)
print(bias_output)

# Fungsi sigmoid digunakan sebagai fungsi aktivasi dalam jaringan, dan turunan fungsi sigmoid diperlukan selama backpropagation.
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

# Proses training
# Melibatkan forward propagation melalui jaringan dan kemudian backpropagation untuk memperbarui bobot dan bias.
epochs = 15000
lr = 0.1

error_list = []
w11_changes = []
w12_changes = []
w21_changes = []
w22_changes = []
w31_changes = []
w32_changes = []
b1_changes = []
b2_changes = [] 
b3_changes = []

for epoch in range(epochs):
    # Forward Propagation
    hiddenLayer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hiddenLayer_output = sigmoid(hiddenLayer_input)

    outputLayer_input = np.dot(hiddenLayer_output, weights_hidden_output) + bias_output
    output = sigmoid(outputLayer_input)

    # Backpropagation
    error = expected_output - output
    error_list.append(np.mean(np.abs(error)))
    d_output = error * sigmoid_derivative(output)
    
    
    error_hiddenLayer = d_output.dot(weights_hidden_output.T)
    d_hiddenLayer = error_hiddenLayer * sigmoid_derivative(hiddenLayer_output)

    # Updating Weights and Biases
    weights_hidden_output += hiddenLayer_output.T.dot(d_output) * lr
    bias_output += np.sum(d_output, axis=0, keepdims=True) * lr
    weights_input_hidden += inputs.T.dot(d_hiddenLayer) * lr
    bias_hidden += np.sum(d_hiddenLayer, axis=0, keepdims=True) * lr
    
    w11_changes.append(weights_input_hidden[0][0])
    w12_changes.append(weights_input_hidden[0][1])
    w21_changes.append(weights_input_hidden[1][0])
    w22_changes.append(weights_input_hidden[1][1])
    w31_changes.append(weights_hidden_output[0][0])
    w32_changes.append(weights_hidden_output[1][0])
    b1_changes.append(bias_hidden[0][0])
    b2_changes.append(bias_hidden[0][1])
    b3_changes.append(bias_output[0][0])

# Figure dan axis instances
fig, ax = plt.subplots()

# Plot scatter - instance
sc = ax.scatter([], [])

# Function to update scatter plot
def update(i):
    ax.clear()
    ax.scatter([i], [error_list[i]])
    ax.scatter([i], [w11_changes[i]])
    ax.scatter([i], [w12_changes[i]])
    ax.scatter([i], [w21_changes[i]])
    ax.scatter([i], [w22_changes[i]])
    ax.scatter([i], [w31_changes[i]])
    ax.scatter([i], [w32_changes[i]])
    ax.scatter([i], [b1_changes[i]])
    ax.scatter([i], [b2_changes[i]])
    ax.scatter([i], [b3_changes[i]])
    ax.set_xlim(0, epochs)
    ax.set_ylim(0, 1)

# Creating animation
ani = animation.FuncAnimation(fig, update, frames=range(epochs), interval=5, repeat=False)

plt.show()
