"""
 Un vector es una entidad matemática que tiene magnitud y dirección.
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Vectores 2D ---
v2 = np.array([3, 4])
w2 = np.array([1, 2])

plt.figure(figsize=(6,6))
ax = plt.gca()

# Función para dibujar vectores 2D
def dibuja_vector2D(vec, color, nombre):
    ax.arrow(0, 0, vec[0], vec[1], 
             head_width=0.2, length_includes_head=True, color=color)
    plt.text(vec[0]+0.2, vec[1]+0.2, nombre, color=color, fontsize=12)

dibuja_vector2D(v2, "blue", "v=(3,4)")
dibuja_vector2D(w2, "red", "w=(1,2)")

plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.5)
plt.title("Vectores 2D")
plt.show()


# --- Vectores 3D ---
v3 = np.array([2, 3, 1])
w3 = np.array([1, -1, 2])

fig = plt.figure(figsize=(7,7))
ax3 = fig.add_subplot(111, projection='3d')

# Función para dibujar vectores 3D
def dibuja_vector3D(vec, color, nombre):
    ax3.quiver(0,0,0, vec[0], vec[1], vec[2], color=color)
    ax3.text(vec[0]+0.2, vec[1]+0.2, vec[2]+0.2, nombre, color=color, fontsize=12)

dibuja_vector3D(v3, "blue", "v=(2,3,1)")
dibuja_vector3D(w3, "red", "w=(1,-1,2)")

ax3.set_xlim([-2, 4])
ax3.set_ylim([-2, 4])
ax3.set_zlim([-2, 4])
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title("Vectores 3D")
plt.show()
