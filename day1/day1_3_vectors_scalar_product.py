import numpy as np
import matplotlib.pyplot as plt

v1 = np.array([1, 2])
v2 = np.array([2, 3])

# skalär produkt - hur lika pilarnas riktingar är (jämföra två linjer/pilar) 
# Dot positiv och stort → pilar nästan parallella
# Dot = 0 → pilar vinkelräta (90°)
# Dot negativ → pilar pekar åt motsatt håll

# v1 * v2 = x1*x2 + y1*y2

dot = np.dot(v1, v2)
print("Dot product:", dot)



# Rita vektorerna från origo
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')

plt.xlim(0, 4)
plt.ylim(0, 4)
plt.grid(True)
plt.legend()
plt.show()
