import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definir vectores ---
v = np.array([3, 4, 23, 34])
w = np.array([1, 2])

# --- 2. Magnitud (norma) ---
# |A| = √(x² + y²) en 2D
# ∥magnitud∥ = √(3^2 + 4^2) = √(9 + 16) = √25 = 5 

magnitud_v = np.linalg.norm(v)
print("\n📏 Magnitud de v:", magnitud_v)


magnitud_w = np.linalg.norm(w)
print("\n📏 Magnitud de w:", magnitud_w)
