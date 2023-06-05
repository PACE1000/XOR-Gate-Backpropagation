#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivatif fungsi aktivasi sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Fungsi untuk melakukan pelatihan dengan backpropagation
def train(X, y, num_epochs):
    np.random.seed(1)
    
    # Inisialisasi bobot secara acak dengan rentang [-1, 1]
    synapse0 = 2 * np.random.random((2, 3)) - 1
    synapse1 = 2 * np.random.random((3, 1)) - 1
    
    for epoch in range(num_epochs):
        # Forward propagation
        layer0 = X
        layer1 = sigmoid(np.dot(layer0, synapse0))
        layer2 = sigmoid(np.dot(layer1, synapse1))
        
        # Menghitung error
        layer2_error = y - layer2
        
        if epoch % 10000 == 0:
            print("Epoch:", epoch, "Error:", np.mean(np.abs(layer2_error)))
        
        # Backpropagation
        layer2_delta = layer2_error * sigmoid_derivative(layer2)
        layer1_error = layer2_delta.dot(synapse1.T)
        layer1_delta = layer1_error * sigmoid_derivative(layer1)
        
        # Update bobot
        synapse1 += layer1.T.dot(layer2_delta)
        synapse0 += layer0.T.dot(layer1_delta)
    
    return synapse0, synapse1

# Fungsi untuk melakukan prediksi dengan model yang telah dilatih
def predict(X, synapse0, synapse1):
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, synapse0))
    layer2 = sigmoid(np.dot(layer1, synapse1))
    return layer2

# Data latih XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T

# Melatih model
synapse0, synapse1 = train(X, y, num_epochs=100000)

# Melakukan prediksi
predictions = predict(X, synapse0, synapse1)
print("Prediksi XOR:")
for i in range(len(X)):
    print(X[i], ":", predictions[i])


# In[ ]:




