# ============================================================
# Ejercicios de Vectores (IA con Python)
# Guardar como 'ejercicios_vectores_python.py' y ejecutar con Python 3
# ============================================================

import numpy as np

# ------------------------
# Funciones útiles
# ------------------------

def to_np(x):
    return np.array(x, dtype=float)

def suma(u, v):
    return to_np(u) + to_np(v)

def escalar(alpha, v):
    return float(alpha) * to_np(v)

def dot(u, v):
    return float(np.dot(to_np(u), to_np(v))) # devuelve un escalar (un número real) que representa el producto punto de dos vectores.

def norma(v):                                # Calcula la norma (longitud) de un vector. 
    return float(np.linalg.norm(to_np(v)))   # cálculo de la longitud (o magnitud) de un vector # 👉 Es el teorema de Pitágoras en varias dimensiones.

def unit(v, eps=1e-12):                      #Normalizar es otra operación: dividir el vector por su norma 
    v = to_np(v)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("No se puede normalizar el vector cero.")
    return v / n

def angulo(u, v, eps=1e-12):
    u = to_np(u); v = to_np(v)
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        raise ValueError("Ángulo indefinido si alguno es vector cero.")
    cos_theta = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.arccos(cos_theta))

def son_ortogonales(u, v, tol=1e-9):
    return abs(dot(u, v)) <= tol

def proyeccion_de_u_sobre_v(u, v, eps=1e-12):
    u = to_np(u); v = to_np(v)
    nv2 = np.dot(v, v)
    if nv2 < eps:
        raise ValueError("No se puede proyectar sobre el vector cero.")
    coef = np.dot(u, v) / nv2
    return coef * v

def componente_perp_de_u_a_v(u, v):
    u = to_np(u)
    return u - proyeccion_de_u_sobre_v(u, v)

def distancia(u, v):
    return float(np.linalg.norm(to_np(u) - to_np(v)))

def cos_sim(u, v, eps=1e-12):
    u = to_np(u); v = to_np(v)
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        raise ValueError("Coseno indefinido si alguno es vector cero.")
    return float(np.dot(u, v) / (nu * nv))

def cross3(u, v):
    return np.cross(to_np(u), to_np(v))

def area_paralelogramo3(u, v):
    return float(np.linalg.norm(cross3(u, v)))

def en_span_de(u, base, tol=1e-9):
    """
    ¿u está en el span (combinación lineal) de los vectores de 'base'?
    """
    U = to_np(u).reshape(-1, 1)
    B = np.column_stack([to_np(b) for b in base])
    coef, residuals, rank, s = np.linalg.lstsq(B, U, rcond=None)
    err = float(np.linalg.norm(B @ coef - U))
    return err <= tol, coef.flatten(), err

# ------------------------
# Ejercicios y pruebas
# ------------------------

if __name__ == "__main__":
#    print("\n=== 1) Operaciones básicas en R^3 ===")
#    u = [1, -2, 3]; v = [4, 0, -1]
#    print("u+v =", suma(u,v))
#    print("3*u =", escalar(3,u))
#    print("dot(u,v) =", dot(u,v)) # multiplicas componente a componente y luego suma los resultados PRODUCTO PUNTO
#    print("|u| =", norma(u))

#    print("\n=== 2) Normalización ===")
#    w = [3,4]
#    print("w =", w)
#    print("||w|| =", norma(w))
#    print("unit(w) =", unit(w))

#   print("\n=== 3) Ángulo entre vectores ===")
#   a = [1,0,0]; b = [1,1,0]
#   theta = angulo(a,b)
#   print("ángulo rad =", theta, " -> deg =", np.degrees(theta))

#     print("\n=== 4) Ortogonalidad (perpendiculares)===")
#     p = [1,2,-1]; q = [2,-1,0]
#     print("dot(p,q) =", dot(p,q))
#     print("¿p ⟂ q? =", son_ortogonales(p,q))
#
#    print("\n=== 5) Proyección y componente perpendicular ===")
#    u = [2,2]; v = [1,0]
#    proj = proyeccion_de_u_sobre_v(u,v)
#    perp = componente_perp_de_u_a_v(u,v)
#    print("proj_v(u) =", proj)
#    print("u_perp_a_v =", perp)
#    print("u =", proj+perp)

#      print("\n=== 6) Distancia ===")
#      x = [1,2,3]; y = [4,6,3]
#      print("dist(x,y) =", distancia(x,y))
#      


#    print("\n=== 7) Similaridad coseno ===")
#    u = [1,0,0,0]; v = [0.9,0.1,0,0]
#    print("cos_sim(u,v) =", cos_sim(u,v))

#   print("\n=== 8) Producto cruz y área ===")
#   u = [1,2,3]; v = [4,5,6]
#   print("u x v =", cross3(u,v))
#   print("|u x v| =", area_paralelogramo3(u,v))


#   print("\n=== 9) ¿Está w en el span de {u,v}? ===")
#   u = [1,0,1]; v = [0,1,1]; w = [2,3,5]
#   esta, coef, err = en_span_de(w,[u,v])
#   print("¿w en span{u,v}? =", esta)
#   print("coeficientes =", coef, "error =", err)



#   print("\n=== 10) Coseno vs Distancia ===")
#   a = [1,2,3,4]; b = [2,4,6,8]
#   print("|a| =", norma(a), "|b| =", norma(b))
#   print("dist(a,b) =", distancia(a,b))
#   print("cos_sim(a,b) =", cos_sim(a,b))


#    print("\n=== 11) Proyección para eliminar componente (bias) ===")
#    vec = [2,3,1]; bias = [1,1,1]
#    proj_bias = proyeccion_de_u_sobre_v(vec,bias)
#    vec_sin_bias = to_np(vec) - proj_bias
#    print("vec_sin_bias =", vec_sin_bias)
#    print("¿vec_sin_bias ⟂ bias? =", son_ortogonales(vec_sin_bias,bias))


#   print("\n=== 12) Base ortonormal en R^2 ===")
#   v1 = [3,4]
#   e1 = unit(v1)
#   e2 = unit([-e1[1], e1[0]])  # rotación 90°
#   print("e1 =", e1)
#   print("e2 =", e2)
#   print("dot(e1,e2) =", dot(e1,e2))
