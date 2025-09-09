import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Exempelvektorer (ändra gärna själv)
v1 = np.array([3, 4])
v2 = np.array([1, 12])

# --- BERÄKNINGAR ---
# skalär produkt - hur lika pilarnas riktingar är (jämföra två linjer/pilar) 
# Dot positiv och stort → pilar nästan parallella
# Dot = 0 → pilar vinkelräta (90°)
# Dot negativ → pilar pekar åt motsatt håll
dot = np.dot(v1, v2)
norm_v1 = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)
cos_sim = dot / (norm_v1 * norm_v2)

# Utskrift dynamiskt
print(f"Dot product ({v1[0]}*{v2[0]} + {v1[1]}*{v2[1]}): {dot}")
print(f"|v1| (sqrt({v1[0]}^2 + {v1[1]}^2)) = {norm_v1}")
print(f"|v2| (sqrt({v2[0]}^2 + {v2[1]}^2)) = {norm_v2}")
print(f"Vectors length product (|v1|*|v2|): {norm_v1 * norm_v2}")
print(f"Cosine similarity formula: {dot} / ({norm_v1} * {norm_v2}) = {cos_sim}")
print(f"Cosine similarity: {cos_sim}")


# --- RITA GRAF ---
plt.figure(figsize=(6,6))
ax = plt.gca()

# Rita vektorer som pilar från origo (0,0)
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label=f"v1 = {v1}")
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label=f"v2 = {v2}")

# Rita koordinatsystem
plt.xlim(min(0, v1[0], v2[0]) - 1, max(0, v1[0], v2[0]) + 1)
plt.ylim(min(0, v1[1], v2[1]) - 1, max(0, v1[1], v2[1]) + 1)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Lägg till etiketter
plt.legend()
plt.grid(True)
plt.title(f"Cosine similarity = {cos_sim:.3f}")
plt.show()
