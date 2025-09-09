
import matplotlib.pyplot as plt
import numpy as np

# Vektorer
v1 = np.array([1, 2])
v2 = np.array([2, 3])

# Rita koordinatsystem
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)

# Rita pilar
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', label='v2')

# Sätt axlar och grid
plt.xlim(0, 3)
plt.ylim(0, 4)
plt.grid(True)
plt.legend()
plt.show()
