import numpy as np
import matplotlib.pyplot as plt

# --- Exempelvektorer ---
v = np.array([3, 4])   # Vektor vi vill projicera
u = np.array([2, 0])   # Riktningen vi projicerar på

# --- Projektion: steg för steg ---
# Skalarprodukt (dot product): v·u = v1*u1 + v2*u2
dot_product = np.dot(v, u)

# |u|^2 = längden av u gånger sig själv (u1^2 + u2^2)
u_length_squared = np.dot(u, u)

# Projektionens formel:
# proj_u(v) = (v·u / |u|^2) * u
proj_v_on_u = (dot_product / u_length_squared) * u

# --- Utskrifter ---
print("=== PROJEKTION AV v PÅ u ===")
print(f"v = {v}")
print(f"u = {u}")
print()
print("FORMEL: proj_u(v) = (v·u / |u|²) * u")
print(f"Dot product v·u = {v[0]}*{u[0]} + {v[1]}*{u[1]} = {dot_product}")
print(f"|u|² = {u[0]}² + {u[1]}² = {u_length_squared}")
print(f"proj_u(v) = ({dot_product} / {u_length_squared}) * {u} = {proj_v_on_u}")
print()

# --- Rita grafen ---
plt.figure(figsize=(6,6))
ax = plt.gca()

# Rita v och u
ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
          color="blue", label=f"v = {v}")
ax.quiver(0, 0, u[0], u[1], angles="xy", scale_units="xy", scale=1,
          color="green", label=f"u = {u}")

# Rita projektionen
ax.quiver(0, 0, proj_v_on_u[0], proj_v_on_u[1], angles="xy", scale_units="xy", scale=1,
          color="red", label=f"proj_u(v) = {proj_v_on_u}")

# Rita en streckad linje mellan v och dess projektion (resten av v som inte pekar i samma riktning)
ax.plot([v[0], proj_v_on_u[0]], [v[1], proj_v_on_u[1]], "k--", label="skillnad")

# Dynamisk skala för både positiva och negativa värden
all_x = [0, v[0], u[0], proj_v_on_u[0]]
all_y = [0, v[1], u[1], proj_v_on_u[1]]

plt.xlim(min(all_x)-1, max(all_x)+1)
plt.ylim(min(all_y)-1, max(all_y)+1)

plt.axhline(0, color='black', linewidth=0.5)  # x-axel
plt.axvline(0, color='black', linewidth=0.5)  # y-axel
plt.grid(True)
plt.legend()
plt.title("Projektion av v på u")
plt.show()
