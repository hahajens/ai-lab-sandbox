import matplotlib.pyplot as plt

v1 = [1, 2]
v2 = [2, 3]

# Rita vektorerna från origo
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')

plt.xlim(0, 4)
plt.ylim(0, 4)
plt.grid(True)
plt.legend()
plt.show()
