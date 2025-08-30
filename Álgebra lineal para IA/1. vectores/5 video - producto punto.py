# 📘 Vectores en IA — Ejemplos prácticos en Python

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definir vectores ---
v = np.array([3, 4])
w = np.array([1, 2])

print("Vector v:", v)
print("Vector w:", w)



"""
    producto punto (dot product) es una operación entre dos vectores
    que da como resultado un escalar (número).
    Se define como la suma de los productos de sus componentes correspondientes.
    Matemáticamente, para dos vectores v y w en R^n:
    v = (v1, v2, ..., vn)
    w = (w1, w2, ..., wn)
    El producto punto se calcula como: v · w = v1*w1 + v2*w2 + ... + vn*

"""
# --- 5. Producto punto (dot product) ---
dot = np.dot(v, w)
print("\n🔹 Producto punto v·w =", dot)
