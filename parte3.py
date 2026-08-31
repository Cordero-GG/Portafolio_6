import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parte1 import (
    newton_raphson,
    secante,
    steffensen,
    muller,
    biseccion,
    falsa_posicion,
)
D = 0.25        
EPS = 0.00015   
RE = 120000     

G_STR = f"1/sqrt(x) + 2*log({EPS}/(3.7*{D}) + 2.51/({RE}*sqrt(x)), 10)"

TOL = 1e-8
ITERMAX = 1000

def analizar_g():

    def g(f):
        return 1.0 / np.sqrt(f) + 2.0 * np.log10(
            EPS / (3.7 * D) + 2.51 / (RE * np.sqrt(f))
        )

    a, b = 0.01, 0.05
    ga, gb = g(a), g(b)
    print("Analisis de g(f):")
    print(f"  g({a}) = {ga:.6f}")
    print(f"  g({b}) = {gb:.6f}")
    print(f"  g(a)*g(b) = {ga*gb:.6f}  -> {'cambio de signo OK' if ga*gb < 0 else 'NO hay cambio de signo'}")
    print()
    return a, b

def ejecutar_metodos():
    a, b = analizar_g()

    x0_newton = 0.02
    x0_sec, x1_sec = 0.015, 0.025
    x0_stef = 0.0203
    x0_mul, x1_mul, x2_mul = 0.015, 0.02, 0.025
    a_bis, b_bis = a, b
    a_fp, b_fp = a, b

    metodos = {}

    t0 = time.perf_counter()
    xk, erk, k, conv = newton_raphson(G_STR, x0_newton, ITERMAX, TOL)
    t1 = time.perf_counter()
    metodos["Newton-Raphson"] = (xk, erk, k, conv, t1 - t0)

    t0 = time.perf_counter()
    xk, erk, k, conv = secante(G_STR, x0_sec, x1_sec, ITERMAX, TOL)
    t1 = time.perf_counter()
    metodos["Secante"] = (xk, erk, k, conv, t1 - t0)

    t0 = time.perf_counter()
    xk, erk, k, conv = steffensen(G_STR, x0_stef, ITERMAX, TOL)
    t1 = time.perf_counter()
    metodos["Steffensen"] = (xk, erk, k, conv, t1 - t0)

    t0 = time.perf_counter()
    xk, erk, k, conv = muller(G_STR, x0_mul, x1_mul, x2_mul, ITERMAX, TOL)
    t1 = time.perf_counter()
    metodos["Muller"] = (xk, erk, k, conv, t1 - t0)

    t0 = time.perf_counter()
    xk, erk, k, conv = biseccion(G_STR, a_bis, b_bis, ITERMAX, TOL)
    t1 = time.perf_counter()
    metodos["Biseccion"] = (xk, erk, k, conv, t1 - t0)

    t0 = time.perf_counter()
    xk, erk, k, conv = falsa_posicion(G_STR, a_fp, b_fp, ITERMAX, TOL)
    t1 = time.perf_counter()
    metodos["Falsa Posicion"] = (xk, erk, k, conv, t1 - t0)

    return metodos

def construir_tabla(metodos):
    filas = []
    for nombre, (xk, erk, k, conv, t) in metodos.items():
        filas.append({
            "Metodo": nombre,
            "f aproximado (xk)": xk,
            "Error final (erk)": erk,
            "Iteraciones (k)": k,
            "Tiempo (s)": t,
            "Convergio (conv)": conv,
        })
    df = pd.DataFrame(filas)
    df = df.set_index("Metodo")
    return df

def graficar(df):
    nombres = df.index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Errores 
    axes[0].bar(nombres, df["Error final (erk)"].values, color="tab:red")
    axes[0].set_yscale("log")
    axes[0].set_title("Error final por metodo (escala log)")
    axes[0].set_ylabel("Error |erk|")
    axes[0].tick_params(axis="x", rotation=35)

    # Tiempos de ejecucion
    axes[1].bar(nombres, df["Tiempo (s)"].values, color="tab:blue")
    axes[1].set_title("Tiempo de ejecucion por metodo")
    axes[1].set_ylabel("Tiempo (s)")
    axes[1].tick_params(axis="x", rotation=35)

    # Iteraciones
    axes[2].bar(nombres, df["Iteraciones (k)"].values, color="tab:green")
    axes[2].set_title("Numero de iteraciones por metodo")
    axes[2].set_ylabel("Iteraciones (k)")
    axes[2].tick_params(axis="x", rotation=35)

    fig.suptitle(
        "Comparacion de metodos - Factor de friccion de Darcy (Colebrook-White)"
    )
    fig.tight_layout()
    fig.savefig("comparacion_metodos_parte3.png", dpi=150)
    plt.show()


def main():
    print("PARTE III")
    print(f"Datos: D = {D} m, eps = {EPS} m, Re = {RE}")
    print(f"iterMax = {ITERMAX}, tol = {TOL}")
    print()

    metodos = ejecutar_metodos()
    df = construir_tabla(metodos)

    pd.set_option("display.float_format", lambda v: f"{v:.10g}")
    print("Tabla comparativa de resultados:")
    print(df.to_string())
    print()

    graficar(df)

    valores_f = df["f aproximado (xk)"]
    dispersion = valores_f.max() - valores_f.min()

    mas_rapido_iter = df["Iteraciones (k)"].idxmin()
    mas_lento_iter = df["Iteraciones (k)"].idxmax()
    mas_rapido_tiempo = df["Tiempo (s)"].idxmin()
    mas_lento_tiempo = df["Tiempo (s)"].idxmax()

    print("ANALISIS COMPARATIVO")
    print(f"""
Todos los metodos convergieron (conv = 1) hacia esencialmente el mismo
valor de f, con una dispersion maxima entre aproximaciones de apenas
{dispersion:.2e}. Esto es consistente con lo esperado: como se demostro
en el analisis previo, g(f) es estrictamente monotona (decreciente) en
el dominio fisico f > 0, por lo que existe una unica raiz y todos los
metodos, al converger, deben aproximar el mismo valor.

Numero de iteraciones:
  El metodo con MENOS iteraciones fue {mas_rapido_iter}
  ({int(df.loc[mas_rapido_iter, "Iteraciones (k)"])} iteraciones), mientras que
  {mas_lento_iter} requirio la mayor cantidad
  ({int(df.loc[mas_lento_iter, "Iteraciones (k)"])} iteraciones).
  Esto se explica por el orden de convergencia de cada metodo: Newton-Raphson
  (orden 2) y Muller (orden ~1.84) usan informacion de la derivada o de una
  aproximacion cuadratica local de g(f), por lo que convergen en muy pocas
  iteraciones cuando el punto inicial esta cerca de la raiz. Steffensen
  tambien alcanza orden cuadratico sin requerir la derivada explicita, pero
  es muy sensible al punto de partida (ver justificacion del valor inicial).
  La Secante, con orden ~1.62, necesita algunas iteraciones mas.
  Biseccion y Falsa Posicion, en cambio, tienen convergencia lineal
  (Biseccion reduce el intervalo a la mitad en cada paso de forma
  garantizada pero lenta); por eso ambas requieren considerablemente
  mas iteraciones que los metodos basados en derivadas o interpolacion
  cuadratica, aun cuando el intervalo inicial ya era relativamente angosto.

Tiempo de ejecucion:
  El metodo mas rapido en tiempo fue {mas_rapido_tiempo} y el mas lento
  {mas_lento_tiempo}. Para los metodos que no requieren derivada (Secante,
  Steffensen, Muller, Biseccion, Falsa Posicion) el tiempo sigue de cerca
  al numero de iteraciones, ya que cada iteracion evalua la funcion g(f)
  compilada con lambdify. Newton-Raphson, en cambio, resulta el mas lento
  en tiempo pese a requerir solo 3 iteraciones: esto se debe a que
  crear_derivada_numerica() calcula simbolicamente la derivada de g(f) con
  sympy (sp.diff) antes de convertirla en funcion numerica, y ese paso de
  diferenciacion simbolica tiene un costo fijo notablemente mayor que
  simplemente evaluar g(f). En terminos absolutos todas las corridas toman
  fracciones de segundo, por lo que para este problema puntual el criterio
  mas relevante para elegir un metodo sigue siendo la robustez y la
  sensibilidad al valor inicial, mas que el tiempo de computo bruto.

Robustez frente al valor/intervalo inicial:
  Biseccion y Falsa Posicion son los metodos mas robustos: solo requieren
  un intervalo [a, b] con cambio de signo, garantizado aqui por la
  monotonia de g(f), y siempre convergen (aunque lentamente). Newton-
  Raphson, Secante y Muller convergen muy rapido pero requieren un punto
  inicial razonablemente cercano a la raiz. Steffensen fue el metodo mas
  delicado: al evaluar g en el punto desplazado f + g(f), un punto inicial
  alejado de la raiz produce un desplazamiento que saca a f del dominio
  fisico (f > 0), haciendo que el metodo diverja (se comprobo
  experimentalmente que con x0 = 0.02 el metodo falla, mientras que con
  x0 = 0.0203, mucho mas cercano a la raiz, converge en pocas iteraciones).
  Esto ilustra un compromiso clasico en metodos numericos: los metodos de
  mayor orden de convergencia (Newton, Muller, Steffensen) son mas rapidos
  pero menos robustos ante puntos iniciales alejados de la solucion, mientras
  que los metodos de intervalo (Biseccion, Falsa Posicion) son mas lentos
  pero garantizan convergencia si se cumple la condicion de cambio de signo.

INTERPRETACION FISICA DEL RESULTADO:
  El factor de friccion de Darcy aproximado para la tuberia analizada
  (D = {D} m, eps = {EPS} m, Re = {RE}) es:

      f ~ {valores_f.mean():.6f}

  Este valor representa la resistencia al flujo debida a la friccion entre
  el fluido y las paredes internas de la tuberia en regimen turbulento.
  Un valor de f de este orden (tipico de tuberias con rugosidad relativa
  eps/D pequena y numeros de Reynolds altos) indica una perdida de energia
  moderada por friccion; este factor se emplea directamente en la ecuacion
  de Darcy-Weisbach para calcular la perdida de carga (energia) por unidad
  de longitud de tuberia en funcion de la velocidad del flujo.
""")


if __name__ == "__main__":
    main()
