"""
Parte II: Comparación computacional de los métodos.

Ecuación: ln(x^2+1) - cos(x) = 4x^2 - 10
Reescrita como g(x) = 0:   g(x) = ln(x^2+1) - cos(x) - 4x^2 + 10
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from parte1 import newton_raphson, secante, steffensen, muller, biseccion, falsa_posicion

FUNCION = "log(x**2+1) - cos(x) - 4*x**2 + 10"
ITER_MAX = 1000
TOL = 1e-8


def g(x):
    """Versión numérica de la ecuación, para explorar y graficar."""
    return np.log(x**2 + 1) - np.cos(x) - 4*x**2 + 10


def justificar_puntos_iniciales():
    """
    Análisis previo de g(x) para elegir los valores iniciales de cada método.

    g(x) = ln(x^2+1) - cos(x) - 4x^2 + 10 es una función PAR (todos sus
    términos son pares: x^2 dentro del log, cos(x), x^2), por lo tanto
    es simétrica respecto al eje y y tiene dos raíces simétricas.

    Explorando g en [-3,3] se observa:
      - g(0) = 9 > 0 (máximo local)
      - g(x) decrece a ambos lados y cruza el cero cerca de x ≈ ±1.69
      - g(-3) ≈ -22.7, g(3) ≈ -22.7 (muy negativo en los extremos)

    Se trabaja con la raíz POSITIVA (x ≈ 1.6938), y se eligen los
    puntos iniciales de cada método alrededor de esa zona:
      - Newton-Raphson, Secante, Steffensen: necesitan puntos cercanos
        a la raíz para converger rápido y de forma estable -> x0=1.5
        (o x0=1.5, x1=2 para Secante)
      - Bisección, Falsa Posición: necesitan un intervalo [a,b] con
        cambio de signo comprobado -> [1, 2], donde g(1)=6.15>0 y
        g(2)=-3.97<0
      - Müller: necesita 3 puntos cercanos a la raíz -> 1.5, 1.7, 1.9
    """
    print("Verificación del intervalo para métodos cerrados:")
    print(f"  g(1) = {g(1):.4f}  (positivo)")
    print(f"  g(2) = {g(2):.4f}  (negativo)")
    print(f"  g(1)*g(2) = {g(1)*g(2):.4f}  -> cambio de signo confirmado, se puede usar [1,2]")
    print()


def correr_metodo(nombre, funcion_llamada):
    """Ejecuta un método midiendo tiempo de ejecución."""
    t0 = time.perf_counter()
    resultado = funcion_llamada()
    t1 = time.perf_counter()
    tiempo = t1 - t0
    xk, erk, k, conv = resultado
    return {
        "metodo": nombre,
        "xk": xk,
        "erk": erk,
        "k": k,
        "tiempo": tiempo,
        "conv": conv,
    }


def main():
    print("=" * 70)
    print("PARTE II: Comparación de métodos")
    print(f"Ecuación: {FUNCION} = 0")
    print("=" * 70)
    print()

    justificar_puntos_iniciales()

    resultados = []

    resultados.append(correr_metodo(
        "Newton-Raphson",
        lambda: newton_raphson(FUNCION, 1.5, ITER_MAX, TOL)
    ))
    resultados.append(correr_metodo(
        "Secante",
        lambda: secante(FUNCION, 1.5, 2, ITER_MAX, TOL)
    ))
    resultados.append(correr_metodo(
        "Steffensen",
        lambda: steffensen(FUNCION, 1.5, ITER_MAX, TOL)
    ))
    resultados.append(correr_metodo(
        "Müller",
        lambda: muller(FUNCION, 1.5, 1.7, 1.9, ITER_MAX, TOL)
    ))
    resultados.append(correr_metodo(
        "Bisección",
        lambda: biseccion(FUNCION, 1, 2, ITER_MAX, TOL)
    ))
    resultados.append(correr_metodo(
        "Falsa Posición",
        lambda: falsa_posicion(FUNCION, 1, 2, ITER_MAX, TOL)
    ))

    # ---- Tabla comparativa ----
    print()
    print("=" * 90)
    print(f"{'Método':<16}{'xk':>16}{'Error':>14}{'Iteraciones':>13}{'Tiempo (s)':>14}{'Conv':>8}")
    print("=" * 90)
    for r in resultados:
        xk_str = f"{r['xk']:.8f}" if r['xk'] is not None else "N/A"
        erk_str = f"{r['erk']:.2e}" if r['erk'] is not None else "N/A"
        k_str = f"{r['k']}" if r['k'] is not None else "N/A"
        print(f"{r['metodo']:<16}{xk_str:>16}{erk_str:>14}{k_str:>13}{r['tiempo']:>14.6f}{r['conv']:>8}")
    print("=" * 90)

    # ---- Gráficas comparativas ----
    metodos = [r["metodo"] for r in resultados]
    errores = [r["erk"] if r["erk"] is not None else 0 for r in resultados]
    tiempos = [r["tiempo"] for r in resultados]
    iteraciones = [r["k"] if r["k"] is not None else 0 for r in resultados]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    axs[0].bar(metodos, errores, color="tab:blue")
    axs[0].set_yscale("log")
    axs[0].set_title("Error final por método")
    axs[0].set_ylabel("Error (escala log)")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(axis='y', alpha=0.3)

    axs[1].bar(metodos, tiempos, color="tab:orange")
    axs[1].set_title("Tiempo de ejecución por método")
    axs[1].set_ylabel("Tiempo (s)")
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(axis='y', alpha=0.3)

    axs[2].bar(metodos, iteraciones, color="tab:green")
    axs[2].set_title("Número de iteraciones por método")
    axs[2].set_ylabel("Iteraciones")
    axs[2].tick_params(axis='x', rotation=45)
    axs[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    # plt.savefig("comparacion_metodos.png", dpi=100)
    plt.show()

    # ---- Análisis comparativo ----
    print()
    print("=" * 70)
    print("ANÁLISIS COMPARATIVO")
    print("=" * 70)

    convergieron = [r for r in resultados if r["conv"] == 1]
    metodo_menos_iter = min(convergieron, key=lambda r: r["k"])
    metodo_menor_tiempo = min(convergieron, key=lambda r: r["tiempo"])

    print(f"""
Los {len(convergieron)} de {len(resultados)} métodos convergieron a la misma raíz
(x ≈ {convergieron[0]['xk']:.6f}), lo cual es consistente porque la ecuación
sí tiene una raíz real única en el intervalo/entorno explorado.

Menor número de iteraciones: {metodo_menos_iter['metodo']} ({metodo_menos_iter['k']} iteraciones).
Esto coincide con la teoría: Newton-Raphson y Müller tienen orden de
convergencia cuadrático (u orden ~1.84 para Müller), mientras que
Bisección tiene convergencia lineal con factor 1/2, necesita muchas
más iteraciones para alcanzar la misma tolerancia.

Menor tiempo de ejecución: {metodo_menor_tiempo['metodo']} ({metodo_menor_tiempo['tiempo']:.6f} s).
El tiempo no siempre sigue el mismo orden que las iteraciones: cada
iteración de Müller es más costosa (resuelve un sistema 3x3 para la
parábola) que una iteración de Bisección (solo evalúa la función una
vez), así que un método con más iteraciones pero trabajo más simple
por iteración puede terminar compitiendo en tiempo total.

Bisección y Falsa Posición, al ser métodos cerrados, garantizan la
convergencia (siempre que el intervalo inicial tenga cambio de signo),
a costa de ser más lentos. Newton-Raphson, Secante, Steffensen y Müller
son métodos abiertos: convergen más rápido cuando el punto inicial está
cerca de la raíz, pero no garantizan la convergencia si el punto inicial
está mal elegido.
""")


if __name__ == "__main__":
    main()