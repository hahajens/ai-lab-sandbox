import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Skapa några "embeddings" i 2D (för enkel visualisering)
# -----------------------------
# points = np.array([
#     [4, 5],
#     [5, 6],
#     [6, 7],
#     [7, 8]
# ])

points = np.array([
    [4, 7],
    [-1, 2],
    [6, -3],
    [0, 0],
    [3, -5]
])

# Varje rad = en "mening", varje kolumn = en dimension

# Beräkna medelvärdet över alla punkter
mean_vector = np.mean(points, axis=0)

# Centrera punkterna
centered_points = points - mean_vector

# -----------------------------
# Plot före centrering
# -----------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.scatter(points[:,0], points[:,1], color='blue')
plt.scatter(mean_vector[0], mean_vector[1], color='red', label='Medelpunkt')
for i, (x, y) in enumerate(points):
    plt.text(x+0.1, y+0.1, f"P{i+1}", fontsize=9)
plt.title("Före centrering")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)

# -----------------------------
# Plot efter centrering
# -----------------------------
plt.subplot(1,2,2)
plt.scatter(centered_points[:,0], centered_points[:,1], color='green')
plt.scatter(0, 0, color='red', label='Origo (medelpunkt)')
for i, (x, y) in enumerate(centered_points):
    plt.text(x+0.1, y+0.1, f"P{i+1}", fontsize=9)
plt.title("Efter centrering")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
