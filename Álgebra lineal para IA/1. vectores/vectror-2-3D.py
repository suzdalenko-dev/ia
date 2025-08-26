import matplotlib.pyplot as plt
import numpy as np

# Vector en R^2
v = np.array([1, 2])  # tomamos solo las dos primeras componentes

plt.figure()
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r')

plt.xlim(0, 2)
plt.ylim(0, 3)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.gca().set_aspect('equal')

plt.show()





# Vector en R^3
v = np.array([15, 11, 3])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dibujar vector como flecha desde el origen hasta (1,2,3)
ax.quiver(0, 0, 0, v[0], v[1], v[2], color='r', arrow_length_ratio=0.1)

# Configurar límites de los ejes
ax.set_xlim([0, 15])
ax.set_ylim([0, 11])
ax.set_zlim([0, 3])

# Etiquetas
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Vector en 3D: (1,2,3)")

plt.show()