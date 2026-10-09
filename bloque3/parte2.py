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
Sí. Los dos métodos tiran la misma distribución de valores propios 
ordenados de mayor a menor. Ambos le llegan al mismo resultado porque analizan el 
mismo sistema tridiagonal.

2. Comportamiento de los residuos
La fórmula  r_j = ||A * v_j - lambda_j * v_j||_2  mide la aproximación.'A' es la matriz, 'v_j' es el vector propio y 'lambda_j' 
su valor propio. La teoría nos dice que sobre un vector propio, la matriz actúa 
como una multiplicación por un escalar (A*v = lambda*v).

La norma 2 (||.||_2) es  medir la distancia de ese error. Por lo general, 
Jacobi tiene residuos más bajos porque sus rotaciones eliminan a ceros los elementos 
fuera de la diagonal sin perder precisión. QR, al ser más pesado, acumula un montón de 
matrices Q, lo que va metiendo basura por los errores de redondeo del sistema.

3. Normalización y Ortogonalidad (er_ort)
Para medir esto usamos  er_ort = ||V^T * V - I||_F. Aquí 'V' son todos los vectores 
propios amontonados y V^T es su transpuesta. 'I' es la matriz identidad. La norma de Frobenius (||.||_F) es como elevar al cuadrado cada
campito de la matriz resultante, sumarlos todos y sacarles raíz. 

Esto nos dice qué tan a 90 grados están los vectores entre sí. Jacobi es mejor aquí 
porque sus rotaciones de Givens mantienen la ortogonalidad. QR vuelve a errar 
un por lo mismom acumula la matriz V tantas veces ensucia la precisión de la máquina.

4. Iteraciones (k) y Tiempos de ejecución
Jacobi obliga a buscar el valor máximo fuera de la diagonal en cada iteración. En una matriz grande 
como esta de 200x200, es muy lento en Python. Por otro lado, QR, aunque haga más iteraciones, se apoya en 
los métodos de NumPy que están optimizados en código C por debajo. Así que QR termina 
el trabajo más rápido y es mejor en tiempo a Jacobi, comprobando por qué usamos métodos que 
trabajan directamente con la matriz para agilizar sistemas inmensos.

5. Criterios de parada
Ambos métodos pararon bajo la misma regla: que la diferencia entre la diagonal actual y 
la anterior sea menor a la tolerancia (er_k < 1e-8). Es decir, el bucle muere cuando a la 
matriz ya se le acabaron los cambios importantes y sus valores fuera de la diagonal 
quedaron siendo unos enanos estadísticos.
"""
    print(analisis)

if __name__ == '__main__':
    run_parte2()