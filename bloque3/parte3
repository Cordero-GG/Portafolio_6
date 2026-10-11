"""
Bloque 3 - Parte III: Reconocimiento facial mediante valores propios
"""

import os
import sys
import time
import textwrap

import numpy as np
import matplotlib

if "--mostrar" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from parte1 import metodo_qr

# Parametros del enunciado

N_PERSONAS = 40
N_ENTRENAMIENTO = 9           # fotografias 1.png ... 9.png por persona
ALTO, ANCHO = 56, 46          # tamano original de cada imagen
N_PIXELES = ALTO * ANCHO      # 2576
ITER_MAX = 10 ** 6
TOL = 1e-5
UMBRAL_RELATIVO = 1e-10       # se conserva lambda_j > 1e-10 * lambda_1
VARIACION_OBJETIVO = 0.95

CARPETA_FIGURAS = "figuras_parte3"
ARCHIVO_PDF = "resultados.pdf"

# Resultados de la Parte II (matriz tridiagonal 200 x 200, iterMax = 10000,
# tol = 1e-8) obtenidos al correr QR y Jacobi de parte1.py. Sirven de base
# para justificar el metodo. Si su parte2.py arroja valores distintos
# (los tiempos dependen de la maquina), reemplacelos aqui.
PARTE2 = {
    "QR":     {"k": 10000, "conv": 0, "rmax": 8.589e-04, "ort": 2.548e-13, "t": 41.6},
    "Jacobi": {"k": 10000, "conv": 0, "rmax": 4.897e-02, "ort": 1.917e-14, "t": 11.4},
}
# Costo medido por iteracion con matrices de 360 x 360.
MS_POR_ITER_QR_360 = 16.0
MS_POR_ITER_JACOBI_360 = 5.0


# Lectura de datos (Pasos 1 a 3)


def buscar_raiz_dataset(base):
    """Devuelve la carpeta que contiene s1, ..., s40 (busca en subcarpetas)."""
    if os.path.isdir(os.path.join(base, "s1")):
        return base
    for actual, subcarpetas, _ in os.walk(base):
        if "s1" in subcarpetas and "s40" in subcarpetas:
            return actual
    raise FileNotFoundError(
        f"No se encontro la base de datos en '{base}'. Descomprima "
        "dataset_att.zip y ejecute:  python parte3.py <carpeta_con_s1_a_s40>"
    )


def cargar_vector(ruta):
    """
    Lee una imagen en escala de grises, la pasa a punto flotante, la divide
    entre 255 y la vectoriza POR COLUMNAS (orden de Fortran), como pide el
    Paso 3.
    """
    imagen = np.asarray(Image.open(ruta).convert("L"), dtype=float) / 255.0
    if imagen.shape != (ALTO, ANCHO):
        raise ValueError(f"{ruta}: se esperaba {ALTO}x{ANCHO} y tiene {imagen.shape}")
    return imagen.flatten(order="F")


def vector_a_imagen(v):
    """Operacion inversa de la vectorizacion por columnas."""
    return np.reshape(v, (ALTO, ANCHO), order="F")


def cargar_base(raiz):
    """
    Construye S (2576 x 360), las etiquetas y de entrenamiento y las
    fotografias de prueba T (2576 x 40) con etiquetas yt. Se respeta el
    orden numerico de las carpetas (s2 antes que s10).
    """
    columnas, etiquetas = [], []
    for persona in range(1, N_PERSONAS + 1):
        for foto in range(1, N_ENTRENAMIENTO + 1):
            columnas.append(cargar_vector(os.path.join(raiz, f"s{persona}", f"{foto}.png")))
            etiquetas.append(persona)
    S = np.column_stack(columnas)
    y = np.array(etiquetas)

    prueba, etiquetas_prueba = [], []
    for persona in range(1, N_PERSONAS + 1):
        prueba.append(cargar_vector(os.path.join(raiz, f"s{persona}", "10.png")))
        etiquetas_prueba.append(persona)
    T = np.column_stack(prueba)
    yt = np.array(etiquetas_prueba)
    return S, y, T, yt



# Utilidades de presentacion

def imagen_para_mostrar(v):
    """Escala un vector (por ejemplo una cara base) a [0, 1] para visualizarlo."""
    img = vector_a_imagen(v)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn) if mx > mn else img


def guardar_figura(fig, nombre, pdf):
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)
    fig.savefig(os.path.join(CARPETA_FIGURAS, nombre), dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")


def paginas_de_texto(pdf, titulo, bloques, lineas_por_pagina=62, ancho=98):
    """Escribe texto en una o varias paginas A4 del PDF."""
    lineas = []
    for bloque in bloques:
        if bloque == "":
            lineas.append("")
        else:
            lineas.extend(textwrap.wrap(bloque, width=ancho, subsequent_indent="   "
                                        if bloque.startswith("-") else ""))
            lineas.append("")
    paginas = [lineas[i:i + lineas_por_pagina] for i in range(0, len(lineas), lineas_por_pagina)] or [[]]
    for n, pagina in enumerate(paginas):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.96, titulo if n == 0 else titulo + " (continuacion)",
                 fontsize=13, fontweight="bold", va="top")
        fig.text(0.07, 0.93, "\n".join(pagina), fontsize=8.6, va="top", family="DejaVu Sans")
        pdf.savefig(fig)
        plt.close(fig)


# Programa principal


def main():
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = argumentos[0] if argumentos else "dataset_att"
    raiz = buscar_raiz_dataset(base)

    pdf = PdfPages(ARCHIVO_PDF)
    salida = []          # texto que se imprime y tambien va al PDF

    def decir(texto=""):
        print(texto)
        salida.append(texto)

    #Pasos 1 a 3: datos 
    S, y, T, yt = cargar_base(raiz)
    print(f"Base cargada desde '{raiz}': S = {S.shape}, imagenes de prueba = {T.shape[1]}")

    #Paso 4: promedio y centrado 
    f_barra = S.mean(axis=1)                 # solo con las 360 de entrenamiento
    A = S - f_barra[:, None]                 # matriz centrada 2576 x 360

    #Paso 5: valores propios y caras base 
    # G = A A^T es de 2576 x 2576. Sus valores propios no nulos coinciden con
    # los de L = A^T A (360 x 360): si L w = lambda w, entonces
    # G (A w) = A (A^T A w) = lambda (A w). Por eso se aplica el metodo QR a L
    # y las caras base se recuperan como u = A w / ||A w||.
    L = A.T @ A
    L = (L + L.T) / 2.0                      # elimina asimetria de redondeo

    print("\nCalculando valores y vectores propios con metodo_qr "
          f"(iterMax = {ITER_MAX:.0e}, tol = {TOL:.0e}) ...")
    t0 = time.perf_counter()
    lam_L, W, er_qr, it_qr, conv_qr = metodo_qr(L, ITER_MAX, TOL)
    t_qr = time.perf_counter() - t0
    lam_L = lam_L.ravel()

    r = int(np.sum(lam_L > UMBRAL_RELATIVO * lam_L[0]))
    lam = lam_L[:r]
    U = A @ W[:, :r]
    U = U / np.linalg.norm(U, axis=0)        # caras base de norma uno

    err_ortogonalidad = np.linalg.norm(U.T @ U - np.eye(r), "fro")
    residuos = np.linalg.norm(A @ (A.T @ U) - U * lam, axis=0)   # ||G u - lambda u||
    residuo_max = residuos.max()

    #Paso 6: numero de caras base 
    variacion_acum = np.cumsum(lam) / np.sum(lam)
    k = int(np.argmax(variacion_acum >= VARIACION_OBJETIVO)) + 1
    variacion_k = variacion_acum[k - 1]
    Uk = U[:, :k]

    #Paso 7: coordenadas de las imagenes conocidas
    X = Uk.T @ A                             # k x 360

    #Paso 8: identificacion de las 40 fotografias de prueba
    coord_prueba = Uk.T @ (T - f_barra[:, None])                      # k x 40
    distancias = np.linalg.norm(coord_prueba.T[:, None, :] - X.T[None, :, :], axis=2)  # 40 x 360
    mas_cercana = np.argmin(distancias, axis=1)      # empate exacto: primera imagen
    predichas = y[mas_cercana]
    correctas = predichas == yt
    n_correctas = int(correctas.sum())
    exactitud = 100.0 * n_correctas / N_PERSONAS

    def nombre_entrenamiento(i):
        return f"s{y[i]}/{i % N_ENTRENAMIENTO + 1}.png"

    # Resultados en terminal (y en el PDF)

    decir("=" * 78)
    decir("1. METODO NUMERICO ESCOGIDO: QR")
    decir("=" * 78)
    p2q, p2j = PARTE2["QR"], PARTE2["Jacobi"]
    justificacion = [
        "Se escogio el metodo QR de la Parte I. Razones, basadas en la Parte II "
        "(matriz tridiagonal 200 x 200, iterMax = 10000, tol = 1e-8):",
        f"- Con el mismo presupuesto de iteraciones, QR alcanzo un residuo maximo "
        f"max ||A v - lambda v|| = {p2q['rmax']:.2e}, frente a {p2j['rmax']:.2e} de Jacobi "
        f"(unas {p2j['rmax'] / p2q['rmax']:.0f} veces mayor). Ninguno de los dos "
        f"declaro convergencia (conv = {p2q['conv']} y {p2j['conv']}), pero QR quedo mucho mas cerca.",
        f"- Ambos conservan la ortogonalidad (||V^T V - I||_F = {p2q['ort']:.1e} en QR y "
        f"{p2j['ort']:.1e} en Jacobi), porque V es un producto de matrices ortogonales, de modo "
        "que la ortogonalidad no distingue a un metodo del otro.",
        f"- Jacobi fue mas barato por iteracion en la Parte II ({p2j['t']:.1f} s frente a {p2q['t']:.1f} s "
        f"para 10000 iteraciones), pero cada iteracion es UNA rotacion y se necesitan del orden de n^2 "
        f"rotaciones. Con n = 360 (matriz de la aplicacion) se midieron unos "
        f"{MS_POR_ITER_JACOBI_360:.0f} ms por rotacion y {MS_POR_ITER_QR_360:.0f} ms por iteracion de QR; "
        "con iterMax = 1e6 el peor caso de Jacobi son horas de calculo sin garantia de alcanzar el residuo de QR.",
        "- El espectro de las imagenes centradas decrece de forma gradual y sin los grupos de valores "
        "propios casi iguales que presenta la matriz tridiagonal de la Parte II, que es lo que frena a QR "
        "sin desplazamiento, y favorece su convergencia con tol = 1e-5 (el resultado real se reporta en la seccion 3).",
        "- Como G = A A^T es de 2576 x 2576, QR y Jacobi se aplican a L = A^T A (360 x 360), "
        "que tiene los mismos valores propios no nulos que G; las caras base se recuperan como "
        "u = A w / ||A w||.",
    ]
    for linea in justificacion:
        decir(textwrap.fill(linea, width=100, subsequent_indent="   " if linea.startswith("-") else ""))

    decir()
    decir("=" * 78)
    decir("2. ROSTRO PROMEDIO Y PRIMERAS SEIS CARAS BASE (ver figuras)")
    decir("=" * 78)
    decir(f"Figuras guardadas en la carpeta '{CARPETA_FIGURAS}' y en {ARCHIVO_PDF}.")

    decir()
    decir("=" * 78)
    decir("3. VALORES PROPIOS RETENIDOS, CARAS BASE Y VARIACION CONSERVADA")
    decir("=" * 78)
    decir(f"Metodo QR sobre L = A^T A (360 x 360): {it_qr} iteraciones, "
          f"er_k = {er_qr:.3e}, conv = {conv_qr}, tiempo = {t_qr:.1f} s")
    decir(f"Valores propios retenidos (lambda_j > 1e-10 lambda_1): r = {r}")
    decir(f"Caras base seleccionadas (menor k con >= 95 % de variacion): k = {k}")
    decir(f"Proporcion de variacion conservada: {100 * variacion_k:.2f} %")
    decir(f"Reduccion de dimension: {N_PIXELES} pixeles -> {k} coordenadas "
          f"(factor {N_PIXELES / k:.1f})")
    decir(f"Comprobacion de las caras base: ||U^T U - I||_F = {err_ortogonalidad:.3e}, "
          f"max ||G u - lambda u|| = {residuo_max:.3e} (relativo a lambda_1: {residuo_max / lam[0]:.3e})")
    decir(f"Primeros valores propios: " + ", ".join(f"{v:.4g}" for v in lam[:6]))

    decir()
    decir("=" * 78)
    decir("4. TABLA DE PREDICCIONES (40 fotografias de prueba)")
    decir("=" * 78)
    cabecera = f"{'Fotografia':<12}{'Persona real':>13}{'Persona predicha':>18}{'Mas cercana':>14}{'Distancia':>11}{'Correcta?':>11}"
    decir(cabecera)
    decir("-" * len(cabecera))
    for t in range(N_PERSONAS):
        decir(f"{f's{yt[t]}/10.png':<12}{yt[t]:>13d}{predichas[t]:>18d}"
              f"{nombre_entrenamiento(mas_cercana[t]):>14}{distancias[t, mas_cercana[t]]:>11.3f}"
              f"{'Si' if correctas[t] else 'No':>11}")

    decir()
    decir("=" * 78)
    decir("5. IDENTIFICACIONES CORRECTAS Y EXACTITUD")
    decir("=" * 78)
    decir(f"Identificaciones correctas: {n_correctas} de {N_PERSONAS}")
    decir(f"Exactitud = 100 * {n_correctas}/{N_PERSONAS} = {exactitud:.2f} %")

    #7. Confusiones 
    decir()
    decir("=" * 78)
    decir("7. CONFUSIONES OBSERVADAS")
    decir("=" * 78)
    confusiones = []
    if n_correctas == N_PERSONAS:
        confusiones.append("No hubo identificaciones incorrectas: las 40 fotografias de prueba "
                           "se asignaron a su persona.")
    else:
        for t in np.where(~correctas)[0]:
            propias = np.where(y == yt[t])[0]
            mejor_propia = propias[np.argmin(distancias[t, propias])]
            d_pred = distancias[t, mas_cercana[t]]
            d_real = distancias[t, mejor_propia]
            confusiones.append(
                f"- s{yt[t]}/10.png (persona {yt[t]}) se identifico como persona {predichas[t]}: la imagen "
                f"mas cercana fue {nombre_entrenamiento(mas_cercana[t])} (distancia {d_pred:.3f}), mientras que "
                f"la mejor imagen de la persona correcta fue {nombre_entrenamiento(mejor_propia)} "
                f"(distancia {d_real:.3f}, {100 * (d_real / d_pred - 1):.1f} % mayor)."
            )
        confusiones.append(
            "- Revise en las figuras de pares las fotografias marcadas en rojo: en este tipo de base "
            "las confusiones suelen aparecer cuando la fotografia de prueba difiere de las de entrenamiento "
            "en pose, expresion, anteojos o iluminacion, y esas diferencias pesan mas en la distancia que "
            "los rasgos propios de la persona."
        )
    for linea in confusiones:
        decir(textwrap.fill(linea, width=100, subsequent_indent="   " if linea.startswith("-") else ""))

    #Analisis final
    analisis = [
        f"El modelo reduce cada fotografia de {N_PIXELES} pixeles a {k} coordenadas y, con una sola "
        f"fotografia de prueba por persona, identifico correctamente {n_correctas} de {N_PERSONAS} "
        f"({exactitud:.2f} %). Con solo 40 pruebas cada error pesa 2.5 puntos, por lo que la exactitud "
        "es una estimacion gruesa del desempeno.",
        f"Papel de los valores propios. La matriz G = A A^T mide como varian conjuntamente los pixeles "
        f"de las imagenes centradas. Cada valor propio lambda_j es la variacion del conjunto de "
        f"entrenamiento a lo largo de la direccion u_j (una cara base). Al ordenarlos de mayor a menor, "
        f"las primeras direcciones concentran la mayor parte de la variacion: con k = {k} de r = {r} "
        f"direcciones se conserva {100 * variacion_k:.2f} % de la variacion, y las {r - k} restantes se "
        "tratan como detalle de poca importancia.",
        f"Papel de los vectores propios. Como G es simetrica, sus vectores propios son ortonormales "
        f"en teoria; numericamente se obtuvo ||U^T U - I||_F = {err_ortogonalidad:.2e}, diferencia que "
        "proviene de detener QR con tol = 1e-5. Forman una base en la que las "
        "coordenadas de cada fotografia se obtienen por simple proyeccion, x = U_k^T (f - f_barra). "
        "Al proyectar, se descartan las direcciones de menor variacion y la distancia euclidiana entre "
        "coordenadas aproxima la distancia entre las imagenes centradas, pero calculada con "
        f"{k} numeros en lugar de {N_PIXELES}.",
        f"Que representa k. El criterio del 95 % se fija solo con las imagenes de entrenamiento y no "
        "equivale a un 95 % de aciertos: conserva variacion de los pixeles, no necesariamente la que "
        "distingue una identidad de otra. Las primeras caras base suelen capturar variaciones globales "
        "(iluminacion, contorno) compartidas por todas las personas, y esa es una posible fuente de "
        "confusiones.",
        f"Sobre el calculo numerico. QR se aplico a L = A^T A (360 x 360) y no a G (2576 x 2576), "
        f"porque ambas tienen los mismos valores propios no nulos y el costo por iteracion es muy "
        f"inferior. Se ejecutaron {it_qr} iteraciones (conv = {conv_qr}, er_k = {er_qr:.2e}, "
        f"{t_qr:.1f} s). La comprobacion sobre G dio un residuo maximo de {residuo_max:.2e} "
        f"({residuo_max / lam[0]:.2e} relativo a lambda_1)."
        + (" El metodo no alcanzo la tolerancia en iterMax, por lo que los vectores propios son "
           "aproximados." if conv_qr == 0 else ""),
        "Limitaciones. Todas las personas de prueba tambien estan en el entrenamiento, de modo que el "
        "procedimiento solo evalua la identificacion entre 40 personas conocidas y no la deteccion o el "
        "rechazo de personas ajenas a la base.",
    ]
    decir()
    decir("=" * 78)
    decir("ANALISIS DE LOS RESULTADOS")
    decir("=" * 78)
    for parrafo in analisis:
        decir(textwrap.fill(parrafo, width=100))
        decir()

    #resultados.pdf
    # (las figuras ya estan en el PDF; se anaden las paginas de texto y la tabla)
    paginas_de_texto(pdf, "Reconocimiento facial mediante valores propios - Resultados", [
        "1. Metodo numerico escogido: QR",
        *justificacion,
        "",
        "3. Valores propios retenidos, caras base y variacion conservada",
        f"- Metodo QR sobre L = A^T A: {it_qr} iteraciones, er_k = {er_qr:.3e}, conv = {conv_qr}, tiempo = {t_qr:.1f} s.",
        f"- Valores propios retenidos r = {r}; caras base seleccionadas k = {k}; variacion conservada = {100 * variacion_k:.2f} %.",
        f"- ||U^T U - I||_F = {err_ortogonalidad:.3e}; max ||G u - lambda u|| = {residuo_max:.3e}.",
        "",
        "5. Identificaciones correctas y exactitud",
        f"- {n_correctas} de {N_PERSONAS} correctas; exactitud = {exactitud:.2f} %.",
    ])

    #Figuras
    # Rostro promedio y seis primeras caras base
    fig, ejes = plt.subplots(2, 4, figsize=(10, 5.6))
    ejes = ejes.ravel()
    ejes[0].imshow(vector_a_imagen(f_barra), cmap="gray", vmin=0, vmax=1)
    ejes[0].set_title("Rostro promedio")
    for j in range(6):
        ejes[j + 1].imshow(imagen_para_mostrar(U[:, j]), cmap="gray")
        ejes[j + 1].set_title(f"Cara base $u_{j + 1}$  ($\\lambda$ = {lam[j]:.3g})", fontsize=9)
    ejes[7].axis("off")
    for ax in ejes[:7]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Rostro promedio y primeras seis caras base (solo entrenamiento)")
    guardar_figura(fig, "rostro_promedio_y_caras_base.png", pdf)

    # Variacion acumulada
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, r + 1), variacion_acum, color="tab:blue", label="Variacion acumulada")
    ax.axhline(VARIACION_OBJETIVO, color="tab:orange", linestyle="--", label="95 %")
    ax.axvline(k, color="tab:red", linestyle=":", label=f"k = {k}")
    ax.plot([k], [variacion_k], "o", color="tab:red")
    ax.set_xlabel("Numero de caras base")
    ax.set_ylabel("Variacion acumulada")
    ax.set_title("Seleccion del numero de caras base")
    ax.legend()
    ax.grid(alpha=0.3)
    guardar_figura(fig, "variacion_acumulada.png", pdf)

    # Pares prueba / mas cercana (dos figuras de 20 pares)
    for parte in range(2):
        fig, ejes = plt.subplots(5, 8, figsize=(12, 8.2))
        for n in range(20):
            t = parte * 20 + n
            fila, col = n // 4, (n % 4) * 2
            color = "green" if correctas[t] else "red"
            ejes[fila, col].imshow(vector_a_imagen(T[:, t]), cmap="gray", vmin=0, vmax=1)
            ejes[fila, col].set_title(f"Prueba s{yt[t]}/10.png", fontsize=6.5, color=color)
            ejes[fila, col + 1].imshow(vector_a_imagen(S[:, mas_cercana[t]]), cmap="gray", vmin=0, vmax=1)
            ejes[fila, col + 1].set_title(f"Cercana {nombre_entrenamiento(mas_cercana[t])}",
                                          fontsize=6.5, color=color)
        for ax in ejes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Fotografias de prueba y fotografia de entrenamiento mas cercana "
                     f"(personas {parte * 20 + 1} a {parte * 20 + 20}; verde = correcta, rojo = incorrecta)",
                     fontsize=10)
        guardar_figura(fig, f"pares_prueba_cercana_{parte + 1}.png", pdf)

    if "--mostrar" in sys.argv:
        plt.show()
    plt.close("all")

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.96, "4. Tabla de predicciones (40 fotografias de prueba)", fontsize=13,
             fontweight="bold", va="top")
    lineas_tabla = [cabecera, "-" * len(cabecera)]
    for t in range(N_PERSONAS):
        lineas_tabla.append(f"{f's{yt[t]}/10.png':<12}{yt[t]:>13d}{predichas[t]:>18d}"
                            f"{nombre_entrenamiento(mas_cercana[t]):>14}{distancias[t, mas_cercana[t]]:>11.3f}"
                            f"{'Si' if correctas[t] else 'No':>11}")
    fig.text(0.07, 0.93, "\n".join(lineas_tabla), fontsize=8, va="top", family="DejaVu Sans Mono")
    pdf.savefig(fig)
    plt.close(fig)

    paginas_de_texto(pdf, "7. Confusiones observadas y analisis de resultados",
                     confusiones + [""] + analisis)
    pdf.close()
    print(f"\nInforme guardado en {ARCHIVO_PDF}")


if __name__ == "__main__":
    main()
