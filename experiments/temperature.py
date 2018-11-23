import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])

# FIXME: Write the code as explained in the instructions

X_scaled = (X / X.min()).reshape([1, -1])
T = np.linspace(0.01, 5, 100).reshape([-1, 1])

P = np.power(X_scaled, -1/T).T / np.sum(np.power(X_scaled, -1/T), axis=1)

print(P)

for i in range(len(X)):
    plt.plot(T, P[i, :], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
