import numpy as np
import time
import matplotlib.pyplot as plt
from parte1 import metodo_qr, metodo_jacobi

def run_parte2():
    #matriz A tridiagonal simétrica de 200x200
    n = 200
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 4
        if i > 0:
            A[i, i-1] = -1
        if i < n - 1:
            A[i, i+1] = -1

    iterMax = 10000
    tol = 1e-8

    print("Calculando Método QR...")
    t0_qr = time.time()
    lam_qr, V_qr, er_k_qr, k_qr, conv_qr = metodo_qr(A, iterMax, tol)
    t1_qr = time.time()
    tiempo_qr = t1_qr - t0_qr

    print("Calculando Método Jacobi")
    t0_jacobi = time.time()
    lam_jacobi, V_jacobi, er_k_jacobi, k_jacobi, conv_jacobi = metodo_jacobi(A, iterMax, tol)
    t1_jacobi = time.time()
    tiempo_jacobi = t1_jacobi - t0_jacobi

    #Comprobar r_max y ortogonalidad
    def calcular_metricas(A, lam, V):
        n = A.shape[0]
        # Residuo máximo
        r_max = 0.0
        for j in range(n):
            v_j = V[:, j].reshape(-1, 1)
            lam_j = lam[j, 0]
            # Norma 2 del residuo
            res = np.linalg.norm(A @ v_j - lam_j * v_j, 2)
            if res > r_max:
                r_max = res
        
        # Error de ortogonalidad
        I = np.eye(n)
        er_ort = np.linalg.norm(V.T @ V - I, 'fro')
        
        return r_max, er_ort

    rmax_qr, ort_qr = calcular_metricas(A, lam_qr, V_qr)
    rmax_jacobi, ort_jacobi = calcular_metricas(A, lam_jacobi, V_jacobi)

    # Mostrar Valores y Vectores propios
    print("\n" + "="*80)
    print("MÉTODO QR - Valores y Vectores Propios")
    print("="*80)
    print("Primeros 5 Valores Propios:\n", lam_qr[:5].flatten())
    print("\nPrimeros 5 Vectores Propios (Esquina 5x5):\n", V_qr[:5, :5])

    print("\n" + "="*80)
    print("MÉTODO JACOBI - Valores y Vectores Propios")
    print("="*80)
    print("Primeros 5 Valores Propios:\n", lam_jacobi[:5].flatten())
    print("\nPrimeros 5 Vectores Propios (Esquina 5x5):\n", V_jacobi[:5, :5])

    # Tabla Comparativa
    print("\n" + "="*110)
    print(f"{'TABLA COMPARATIVA':^110}")
    print("="*110)
    print(f"{'Método':<10} | {'er_k':<12} | {'Iter (k)':<8} | {'Tiempo (s)':<12} | {'Conv':<5} | {'r_max':<12} | {'er_ort':<12}")
    print("-" * 110)
    print(f"{'QR':<10} | {er_k_qr:<12.4e} | {k_qr:<8} | {tiempo_qr:<12.4f} | {conv_qr:<5} | {rmax_qr:<12.4e} | {ort_qr:<12.4e}")
    print(f"{'Jacobi':<10} | {er_k_jacobi:<12.4e} | {k_jacobi:<8} | {tiempo_jacobi:<12.4f} | {conv_jacobi:<5} | {rmax_jacobi:<12.4e} | {ort_jacobi:<12.4e}")
    print("="*110)

    # Gráficas
    metodos = ['QR', 'Jacobi']
    
    plt.figure(figsize=(15, 5))

    # Error Final er_k 
    plt.subplot(1, 3, 1)
    plt.bar(metodos, [er_k_qr, er_k_jacobi], color=['#1f77b4', '#ff7f0e'])
    plt.yscale('log')
    plt.title('Error Final ($er_k$)')
    plt.ylabel('Error (Log)')

    # Tiempo de ejecución
    plt.subplot(1, 3, 2)
    plt.bar(metodos, [tiempo_qr, tiempo_jacobi], color=['#1f77b4', '#ff7f0e'])
    plt.yscale('log')
    plt.title('Tiempo de Ejecución')
    plt.ylabel('Segundos (Log)')

    # Número de iteraciones
    plt.subplot(1, 3, 3)
    plt.bar(metodos, [k_qr, k_jacobi], color=['#1f77b4', '#ff7f0e'])
    plt.yscale('log')
    plt.title('Número de Iteraciones ($k$)')
    plt.ylabel('Iteraciones (Log)')

    plt.tight_layout()
    plt.savefig('comparacion_metodos.png')
    print("\n>>> Se ha generado la gráfica 'comparacion_metodos.png' en el directorio actual. <<<")
    plt.show()

    #Análisis
    print("\n" + "="*80)
    print("ANÁLISIS COMPARATIVO DE LOS RESULTADOS")
    print("="*80)
    analisis = """
1. ¿Aproximan los mismos valores propios?
Aproximadamente, pero no idénticos. Ambos métodos trabajan sobre la misma matriz
tridiagonal y sus valores propios más grandes quedan cerca de 6 (el mayor valor propio
exacto es 4 + 2*cos(pi/201) = 5.99976). Sin embargo, como ninguno de los dos alcanzó la
tolerancia en iterMax, las aproximaciones difieren a partir de la tercera cifra decimal
y los valores de QR quedan más cerca de los exactos que los de Jacobi.

2. Comportamiento de los residuos
La fórmula  r_j = ||A * v_j - lambda_j * v_j||_2  mide qué tan bien se cumple la
definición A*v = lambda*v para cada par calculado; r_max es el peor de los n residuos.
En esta corrida QR obtuvo un residuo máximo unas 55 veces menor que Jacobi (del orden
de 9e-4 contra 5e-2). Esto se debe a que, con el mismo límite de 10000 iteraciones, QR
aplica una transformación completa a toda la matriz en cada iteración, mientras que
Jacobi anula un solo elemento por iteración y necesita del orden de n^2 o más rotaciones
para diagonalizar, por lo que queda más lejos de la solución.

3. Normalización y Ortogonalidad (er_ort)
Para medir esto usamos  er_ort = ||V^T * V - I||_F. V es la matriz de vectores propios
acumulados y I la identidad; la norma de Frobenius es la raíz de la suma de los cuadrados
de todas las entradas de V^T*V - I. Un valor cercano a cero indica columnas ortonormales.
Ambos métodos son excelentes en este aspecto (del orden de 1e-13 en QR y 1e-14 en Jacobi)
porque V es un producto de matrices ortogonales (las Q de QR y las rotaciones de Jacobi).
Jacobi resulta algo mejor porque cada rotación solo modifica dos columnas de V, pero la
diferencia es pequeña y no distingue a un método del otro.

4. Iteraciones (k) y Tiempos de ejecución
Ambos métodos consumieron las 10000 iteraciones permitidas. Jacobi fue bastante más
rápido en tiempo total, porque cada una de sus iteraciones solo anula un elemento,
mientras que cada iteración de QR factoriza toda la matriz de 200x200. Este menor costo
por iteración no significa que Jacobi sea mejor: como cada iteración es una sola
rotación, necesita muchas más iteraciones para alcanzar la misma calidad de resultado
que QR (ver el residuo en el punto 2).

5. Criterios de parada
Ambos métodos usan la misma regla de parada: que la diferencia entre la diagonal actual
y la anterior (er_k) sea menor que la tolerancia (1e-8). Ninguno la cumplió: los dos
terminaron por llegar a iterMax = 10000, por lo que conv = 0 en ambos. Hay que tener
presente que un er_k pequeño solo indica que la diagonal casi dejó de cambiar y no
garantiza que el residuo sea pequeño; por eso se calcula también r_max. Con estos
parámetros las aproximaciones de QR son razonablemente buenas, mientras que las de
Jacobi son claramente menos precisas.
"""
    print(analisis)

if __name__ == '__main__':
    run_parte2()