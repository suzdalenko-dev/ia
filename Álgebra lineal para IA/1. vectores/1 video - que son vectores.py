# 📘 Vectores en IA — Ejemplos prácticos en Python

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definir vectores ---
v = np.array([3, 4])
w = np.array([1, 2])

print("Vector v:", v)
print("Vector w:", w)



# # --- 6. Visualización ---
# plt.figure(figsize=(6,6))
# ax = plt.gca()
# 
# # Dibujar vectores
# def dibuja_vector(vec, color, nombre):
#     ax.arrow(0, 0, vec[0], vec[1], 
#              head_width=0.2, length_includes_head=True, color=color)
#     plt.text(vec[0]+0.1, vec[1]+0.1, nombre, color=color, fontsize=12)
# 
# dibuja_vector(v, "blue", "v")
# dibuja_vector(w, "green", "w")
# dibuja_vector(suma, "red", "v+w")
# 
# # Ejes y estilo
# plt.xlim(-1, 6)
# plt.ylim(-1, 6)
# plt.axhline(0, color="black", linewidth=0.5)
# plt.axvline(0, color="black", linewidth=0.5)
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.title("Vectores v, w y su suma")
# plt.show()
