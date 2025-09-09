import matplotlib.pyplot as plt
import numpy as np

y = np.linspace(0, 1, 100)
phi = -0.9 * np.exp(-10 * y)
ne = np.exp(phi)
ni = 1 / np.sqrt(1 + 2 * phi)

plt.figure()
plt.plot(y, phi, label="$\\phi$")
plt.xlabel("$y$")
plt.ylabel("$\\phi$")

plt.figure()
plt.plot(y, ne, label="$n_e$")
# plt.plot(y, ni, label="$n_i$")
plt.legend()
plt.xlabel("$y$")
plt.ylabel("$n$")
plt.show()
