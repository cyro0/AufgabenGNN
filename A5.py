import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def neuron(o, wbias, w):
  net_input = wbias + np.dot(w, o)
  return tanh(net_input)

# Initialisierung der Parameter
wbias1 = -3.37
wbias2 = 0.125
w11 = -4
w12 = 1.5
w21 = -1.5
w22 = 0

# Initialisierung der Anfangszustände
o1 = 0.0
o2 = 0.0

# Festlegen der Anzahl an Zeitschritten
num_steps = 100

# Speichern der neuronalen Ausgaben in Listen
o1_history = []
o2_history = []

# Simulation des rekursiven neuronalen Netzes
for t in range(num_steps):
  o = np.array([o1, o2])
  o_new = neuron(o, [wbias1, wbias2], [[w11, w12], [w21, w22]])
  o1 = o_new[0]
  o2 = o_new[1]

  o1_history.append(o1)
  o2_history.append(o2)

# Plotten der neuronalen Ausgaben
plt.plot(o1_history, label='o1')
plt.plot(o2_history, label='o2')
plt.xlabel('Zeitschritt')
plt.ylabel('Neuronales Ausgabe')
plt.title('Rekurrentes neuronales Netz mit zwei Neuronen')
plt.legend()
plt.show()
