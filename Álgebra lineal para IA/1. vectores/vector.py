import numpy as np

"""
    🔹 ¿Qué es un vector?
        y no en mates, que es un vector?
        Un vector es un objeto matemático que tiene magnitud y dirección.
        
        En mates, un vector es un objeto que tiene:
        Magnitud (qué tan grande es).
        Dirección (hacia dónde apunta)

    1.  La Magnitud de un vector es su "longitud" o "tamaño" (representacion del valor) lo que mide o representa ese vector.
        La Dirección de un vector es la orientación o el camino que sigue ese vector en el espacio
        (en 2D puede ser un ángulo respecto al eje x, en 3D puede ser más complejo).
        Un vector se representa como una flecha en un espacio (2D, 3D, o más dimensiones).
        Un vector en 2D se representa como v = (x, y) donde x es la componente horizontal y y es la componente vertical.
        En 3D se representa como v = (x, y, z) donde z es la componente en profundidad.
        En espacios de más dimensiones, se representa como una lista de números v = (x1, x2, x3, ..., xn).
        |A| = √(x² + y²) en 2D
        |A| = √(x² + y² + z²) en 3
        |A| = √(x1² + x2² + ... + xn²) en n dimensiones
        Un vector puede tener cualquier número de dimensiones, desde 2D y 3D hasta espacios de muchas dimensiones.
        Un vector es diferente de un escalar, que es solo un número sin dirección (por ejemplo, temperatura o masa).

        Calculo de la magnitud:
        La magnitud de un vector v = (x, y) en 2D se calcula con la fórmula:
        v = (3, 4)
        |v| = √(3² + 4²) = √(9 + 16) = √25 = 5
        v2 = (-5, 2)
        |v2| = √((-5)² + 2²) = √(25 + 4) = √29 ≈ 5.39

    2. Un escalar es solo un número que representa una cantidad sin dirección (por ejemplo, temperatura o masa).
       Un vector tiene tanto magnitud como dirección (por ejemplo, velocidad o fuerza).    
        👉 Ejemplo físico: velocidad. Si digo “80 km/h hacia el norte”, 
        no basta con el número 80 (eso sería un escalar), también necesito la dirección → eso es un vector.

        👉 Ejemplo más de calle: cuando usas Google Maps y te muestra una flecha azul, eso es un vector: apunta a dónde ir, 
        y puede tener magnitud (distancia).

    Como se representa:
        En papel es una lista de numeros en 2D v=[3, 4] segnifica 3 pasos a la derecha eje x y 4 pasos hacia arriba eje y
        En 3D v=[1, 2, -5]
        1 paso en x, 2 pasos en y, 5 pasos hacia abajo en z

    En la vida real puede tener miles de dimensiones, por ejemplo:
        un vector de un cliente de banco:
            [edad, ingresos, numero compras, saldo, valor prestado]

    
    


            
    🔹 Tipos de vectores (según el contexto)
        fila [1,2,3]    (tipo vector horizontal tiene una fila  y 3 columnas)
        columna [1      (tipo vector vertical tiene 3 filas y 1 columna)
                 2
                 3]    
        La diferencia no es el "objeto matematico" en si, si no como lo usamos dentro de operaciones matriciales,
        piensa que un vector en un caso especial de matriz:
            - Vector fila    = matriz de una fila
            - Vector columna = matriz de una columna
    
        Ejemplo por que importa:
                      [4
            [1 2 3] *  5 = 1*4 + 2*5 + 3*6 = 32
                       6]
            👉 Da un escalar (1 numero). Esto es producto punto

          Si es al reves:
          [4             [4 8  12
           5 * [1 2 3] =  5 10 15
           6]             6 12 18]
            👉 Da una matriz 3×3 (producto exterior).

        




    REGLA BASICA, un vector es una lista (array) de numeros
    Un vector en R2 se esribe v=(x, y)
    El primer componente es el x horizontal
    El segundo componente es el y vertical
    de aqui
    v=(1,2) => x=1, y=2

    2∈{1,2,3} ✅ (2 pertenece al conjunto).
    {1,2}⊂{1,2,3} ✅ (el conjunto {1,2} está contenido en {1,2,3}).
"""


# 2 listas (vectores)
vector1 = [1,2,3,6,3,65,67,2]
vector2 = [11,23,31,61,31,654,167,12]

# vectores en python
v1 = np.array([1,2,3,6,3,65,67,2])
v2 = np.array([11,23,31,61,31,654,167,12])


"""
Regla básica:
Dos vectores solo se pueden sumar si tienen la misma dimensión (misma cantidad de componentes).
"""

a = np.array([1,2,3])
b = np.array([4,5])
# print(a + b)   # ValueError: operands could not be broadcast together with shapes (3,) (2,) 

a = np.array([1,2,3])
b = np.array([4,5, 111])

# 0. SUMA
print(a + b)  

# 1. RESTA
print(a - b)   # resta componente a componente

# 2. Multiplicación por un escalar (estirar/encoger)
print(3 * a)   # multiplica cada componente por 3
print(-2 * b)  # cambia signo y estira

# 3. Producto punto (dot product) → mide “cuánto apuntan en la misma dirección”
# a⋅b=1⋅4+2⋅5+3⋅111       = 347
print(np.dot(a, b))


# 4. Norma o longitud del vector (distancia al origen)
# ∥a∥=12+22+32          = 3.7416573867739413
print(np.linalg.norm(a)) 


# 5. Distancia entre vectores
# ∥a−b∥                 = 108.08330120791094
print(np.linalg.norm(a - b))

# 6. Producto cruzado (en 𝑅3) → genera un vector perpendicular a ambos
print(np.cross(a, b))  # = [207 -99  -3]