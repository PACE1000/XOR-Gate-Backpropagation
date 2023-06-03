import numpy as npi
import matplotlib.pyplot as plot
import matplotlib.animation as animation

# Menentukan kombinasi input dan output pada gerbang XOR
x = npi.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = npi.array([[0], [1], [1], [0]])
# Menentukan parameter
learning = 0.2
epoc = 5000
dim_input = 2
dim_output = 1
hidden = 2
# random weight dan bias pada layer hidden
npi.random.seed(0)
w1 = npi.random.randn(dim_input, hidden)
b1 = npi.zeros((1, hidden))
w2 = npi.random.randn(hidden, dim_output)
b2 = npi.zeros((1, dim_output))
# Fungsi Sigmoid
def sigmoid(x):
    return 1 / (1 + npi.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize the figure and axis
fig, ax = plot.subplots()
ax.set_xlim(0, epoc)
ax.set_ylim(0, 0.5)
line, = ax.plot([], [], lw=2)

# Initialize the loss data
losses = []
epoch_values = []

# Update function for the animation
def update(epoch):
    global w1, b1, w2, b2

    # Forward propagation
    hidden_layer_output = sigmoid(npi.dot(x, w1) + b1)
    output = sigmoid(npi.dot(hidden_layer_output, w2) + b2)

    # Calculate the loss (mean squared error)
    loss = npi.mean((output - y) ** 2)
    losses.append(loss)
    epoch_values.append(epoch)

    # Backpropagation
    error = output - y
    delta_output = error * sigmoid_derivative(npi.dot(hidden_layer_output, w2) + b2)
    delta_hidden = npi.dot(delta_output, w2.T) * sigmoid_derivative(npi.dot(x, w1) + b1)

    # Update the weights and biases
    w2 -= learning * npi.dot(hidden_layer_output.T, delta_output)
    b2 -= learning * npi.sum(delta_output, axis=0)
    w1 -= learning * npi.dot(x.T, delta_hidden)
    b1 -= learning * npi.sum(delta_hidden, axis=0)

    # Update the line data
    line.set_data(epoch_values, losses)

    # Check if epoch exceeds 2000
    if epoch >= 5000:
        ani.event_source.stop()

    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(epoc), interval=1, blit=True)

# Display the animation
plot.xlabel('Epoch')
plot.ylabel('Loss')
plot.title('Grafik Perbandingan Loss dan Epochs')
plot.show()
