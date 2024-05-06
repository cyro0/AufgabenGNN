import numpy as np

# Sigmoide Aktivierungsfunktion und ihre Ableitung
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # Sigmoidfunktion

def deriv_sigmoid(x):
    return x * (1 - x) # Ableitung der Sigmoiden

# Das XOR-Problem, input [bias, x, y] und Target-Daten
inp    = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
target = np.array([[0], [1], [1], [0]])

# Die Architektur des neuronalen Netzes
inp_size = 3   # Eingabeneuronen
hid_size = 4   # Hidden-Neuronen
out_size = 1   # Ausgabeneuron
eta_plus = 1.2  # Faktor zur Vergrößerung der Lernrate
eta_minus = 0.5 # Faktor zur Verkleinerung der Lernrate
delta_max = 50 # Maximale Gewichtsänderung
delta_min = 0  # Minimale Gewichtsänderung

# Gewichte zufällig initialisieren
weightLayers = [
    np.random.random((inp_size, hid_size)) - 0.5,
    np.random.random((hid_size, out_size)) - 0.5
]

# Initialisiere die Gewichtungsänderungen für alle Verbindungen im Netzwerk mit dem Wert 0.5, um Trainingszeit zu beeinflussen
deltaLayers = [
    np.array([hid_size * [0.5]] * inp_size),
    np.array([out_size * [0.5]] * hid_size)
]

# Initialisiere Gradienten der Fehlerfunktion in Bezug auf die Gewichte aus der vorherigen Iteration
prevGradientLayers = [
    np.zeros((inp_size, hid_size)),
    np.zeros((hid_size, out_size))
]

# Netzwerk trainieren
for i in range(10000):
    layers = [[], [], []]
    layers[0] = inp

    # Vorwärtsaktivierung
    for i in range(len(weightLayers)):
        layers[i][0] = 1 # Bias-Neuron Hiddenschicht
        layers[i + 1] = sigmoid(np.matmul(layers[i], weightLayers[i]))

    # Fehler berechnen
    error = layers[2] - target

    
    for i in range(len(weightLayers)):
        deltaW = deltaLayers[1 - i]
        prev_gradient = prevGradientLayers[1 - i]
        gradient = np.matmul(layers[1 - i].T, error) # Berechne die Gradienten der Fehlerfunktion

        # Implementiere Teil des iRprop-Algorithmus, der zur Anpassung der Lernrate für die Gewichtungsaktualisierungen verwendet wird.
        for j in range(len(gradient)):
            for k in range(len(gradient[j])):
                #Lernrate vergrößern wenn Vorzeichen gleich bleibt
                if prev_gradient[j][k] * gradient[j][k] > 0:
                    deltaW[j][k] = min(deltaW[j][k] * eta_plus, delta_max)
                
                #Lernrate verkleinern wenn Vorzeichen sich ändert
                if prev_gradient[j][k] * gradient[j][k] < 0:
                    deltaW[j][k] = max(deltaW[j][k] * eta_minus, delta_min)
                    gradient[j][k] = 0 # Unterschied zu Rprop
        prevGradientLayers[1 - i] = gradient
        weightLayers[1 - i] -= deltaW * np.sign(gradient)

        delta = error * deriv_sigmoid(layers[2 - i]) # Die Größe und Richtung der Aktualisierung wird durch die Schrittweite (deltaW) und dem Vorzeichen des Gradienten bestimmt.
        error = np.matmul(delta, weightLayers[1 - i].T) # Berechne Fehler des vorherigen Schritts
                
# Netzwerk testen
L0 = inp
L1 = sigmoid(np.matmul(inp, weightLayers[0]))
L1[0] = 1 # Bias-Neuron in der Hiddenschicht 
L2 = sigmoid(np.matmul(L1, weightLayers[1]))

print(L2)