import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Inisialisasi tabel kebenaran
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

# Buat fungsi sigmoid dan turunannya
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def calculate_error(y, y_hat):
    return 0.5 * np.mean((y - y_hat)**2)

# Berdasarkan ilustrasi gerbang XOR
inputLayer = 2
hiddenLayer = 2
outputLayer = 1

# Inisialisasi weight dan bias dengan nilai random
hidden_weights = np.random.uniform(size=(inputLayer,hiddenLayer))
hidden_bias =np.random.uniform(size=(1,hiddenLayer))

output_weights = np.random.uniform(size=(hiddenLayer,outputLayer))
output_bias = np.random.uniform(size=(1,outputLayer))

# print("Weight awal untuk hidden layer: ",end='')
# print(*hidden_weights)
# print("Bias awal untuk hidden layer: ",end='')
# print(*hidden_bias)
# print("Weight awal untuk output layer: ",end='')
# print(*output_weights)
# print("Bias awak untuk output layer: ",end='')
# print(*output_bias)

# Buat list untuk tiap perubahannya
error_changes = []
w11_changes = []
w12_changes = []
w21_changes = []
w22_changes = []
w31_changes = []
w32_changes = []
b1_changes = []
b2_changes = []
b3_changes = []

# Tentukan learning rate dan epoch
lr = 0.1
epoch = 100

# Training algorithm
for g in range(epoch):
	# Forward pass
	hidden_layer_activation = np.dot(inputs,hidden_weights)
	hidden_layer_activation += hidden_bias
	hidden_layer_output = sigmoid(hidden_layer_activation)

	output_layer_activation = np.dot(hidden_layer_output,output_weights)
	output_layer_activation += output_bias
	predicted_output = sigmoid(output_layer_activation)

	# Hitung nilai error
	error = calculate_error(expected_output, predicted_output)
	error_changes.append(error)

	# Backward pass
	delta_output = expected_output - predicted_output
	delta_predicted_output = delta_output * sigmoid_derivative(predicted_output)
	
	error_hidden_layer = delta_predicted_output.dot(output_weights.T)
	delta_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

	# Ubah nilai weight and bias
	output_weights += hidden_layer_output.T.dot(delta_predicted_output) * lr
	output_bias += np.sum(delta_predicted_output,axis=0,keepdims=True) * lr
	hidden_weights += inputs.T.dot(delta_hidden_layer) * lr
	hidden_bias += np.sum(delta_hidden_layer,axis=0,keepdims=True) * lr

	# Simpan perubahan ke list yang tersedia
	w11_changes.append(hidden_weights[0][0])
	w12_changes.append(hidden_weights[0][1])
	w21_changes.append(hidden_weights[1][0])
	w22_changes.append(hidden_weights[1][1])
	w31_changes.append(output_weights[0][0])
	w32_changes.append(output_weights[1][0])
	b1_changes.append(hidden_bias[0][0])
	b2_changes.append(hidden_bias[0][1])
	b3_changes.append(output_bias[0][0])

# Inisialisasi grafik
fig, ax = plt.subplots()
ax.set_xlim(0, len(w11_changes))
ax.set_ylim(0, 4)
ax.set_xlabel('Perubahan Nilai Weight')
ax.set_ylabel('Error')
ax.set_title('Perubahan Nilai Weight dan Bias sepanjang Nilai Error')

# Inisialisasi scatter plot 
# Plot untuk menunjukkan perubahan weight dan bias
sc_w11 = ax.scatter([], [], c='#257A25', label='w11')
sc_w12 = ax.scatter([], [], c='#CC7D1D', label='w12')
sc_w21 = ax.scatter([], [], c='#9E0B0B', label='w21')
sc_w22 = ax.scatter([], [], c='#6DE4E8', label='w22')
sc_w31 = ax.scatter([], [], c='#1E2387', label='w31')
sc_w32 = ax.scatter([], [], c='#BB7EED', label='w32')
sc_b1 = ax.scatter([], [], c='#7EF77E', label='b1')
sc_b2 = ax.scatter([], [], c='#F2ED61', label='b2')
sc_b3 = ax.scatter([], [], c='#A80D84', label='b3')
ax.legend() # Buat legenda

# Inisialisasi plot grafik fungsi error
line, = ax.plot([], [], c='black', label='error')
ax.legend() # Masukkan juga ke legenda

def update_scatter_plot(i):
    # Mengambil data perubahan weight pada indeks i
    w11_val = w11_changes[i]
    w12_val = w12_changes[i]
    w21_val = w21_changes[i]
    w22_val = w22_changes[i]
    w31_val = w31_changes[i]
    w32_val = w31_changes[i]
    b1_val = b1_changes[i]
    b2_val = b2_changes[i]
    b3_val = b3_changes[i]
    error_val = error_changes[i]

    # Mengatur posisi pada masing-masing scatter plot
    sc_w11.set_offsets(np.array([[i, w11_val], [i, error_val]]))
    sc_w12.set_offsets(np.array([[i, w12_val], [i, error_val]]))
    sc_w21.set_offsets(np.array([[i, w21_val], [i, error_val]]))
    sc_w22.set_offsets(np.array([[i, w22_val], [i, error_val]]))
    sc_w31.set_offsets(np.array([[i, w31_val], [i, error_val]]))
    sc_w32.set_offsets(np.array([[i, w32_val], [i, error_val]]))
    sc_b1.set_offsets(np.array([[i, b1_val], [i, error_val]]))
    sc_b2.set_offsets(np.array([[i, b2_val], [i, error_val]]))
    sc_b3.set_offsets(np.array([[i, b3_val], [i, error_val]]))

    # Memperbarui data pada plot grafik fungsi error
    line.set_data(range(i+1), error_changes[:i+1])

    return sc_w11, sc_w12, sc_w21, sc_w22, sc_w31, sc_w32, sc_b1, sc_b2, sc_b3, line

# Animasi scatter plot
ani = animation.FuncAnimation(fig, update_scatter_plot, frames=len(w11_changes), interval=200, blit=True)
plt.show()