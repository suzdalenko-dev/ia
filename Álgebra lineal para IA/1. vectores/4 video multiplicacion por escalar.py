# 📘 Vectores en IA — Ejemplos prácticos en Python

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definir vectores ---
v = np.array([3, 4])
w = np.array([1, 2])

print("Vector v:", v)
print("Vector w:", w)


# --- 4. Producto por escalar ---
escalar = 2
escalar_v = escalar * v
print(f"\n✖️ {escalar} * v =", escalar_v)
