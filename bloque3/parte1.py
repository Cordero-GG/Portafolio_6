import numpy as np

# =====================================================================
# Funcion auxiliar interna (compartida por ambos metodos)
# =====================================================================

def _ordenar_y_normalizar(Ak, V):
    """
    Toma la matriz Ak (ya diagonal o casi-diagonal, segun el metodo
    que la produjo) y la matriz de vectores propios acumulados V, y:

        1. extrae lam = diag(Ak),
        2. ordena lam de mayor a menor, reordenando las columnas de
           V para que sigan correspondiendo a los mismos vectores,
        3. normaliza cada columna de V (seguridad numerica ante
           errores de redondeo acumulados),
        4. entrega lam como vector columna (n x 1), como lo exige
           el enunciado.

    Se factoriza aqui porque metodo_qr y metodo_jacobi necesitan
    exactamente el mismo post-procesamiento antes de retornar.
    """
    n = Ak.shape[0]

    lam = np.diag(Ak).copy()
    idx = np.argsort(-lam) # orden descendente
    lam = lam[idx]
    V = V[:, idx]

    for j in range(n):
        V[:, j] = V[:, j] / np.linalg.norm(V[:, j])

    return lam.reshape(n, 1), V


# =====================================================================
# 1. Metodo QR
# =====================================================================

def metodo_qr(A, iterMax, tol):
    """
    Calcula los valores y vectores propios de una matriz simetrica A
    mediante el Metodo QR visto en clase.

    Idea del metodo:
        Se construye la sucesion A_0 = A, A_{k+1} = R_k * Q_k, donde
        A_k = Q_k * R_k es la factorizacion QR de A_k en la iteracion
        k (se usa np.linalg.qr para esta factorizacion, ya que el
        enunciado permite apoyarse en la factorizacion QR de
        biblioteca dentro de un ciclo iterativo programado
        manualmente). Como A es simetrica, A_k converge a una matriz
        diagonal cuyas entradas son los valores propios de A.

        Los vectores propios se obtienen acumulando las matrices Q
        de cada iteracion: V_{k+1} = V_k * Q_k, con V_0 = I.

    Criterio de parada:
        er_k = || diag(A_k) - diag(A_{k-1}) ||_2 < tol

    Parametros
    ----------
    A : ndarray (n x n)
        Matriz simetrica cuyos valores y vectores propios se buscan.
    iterMax : int
        Numero maximo de iteraciones permitidas.
    tol : float
        Tolerancia para el criterio de parada.

    Retorna
    -------
    lam  : ndarray (n x 1) -- valores propios, de mayor a menor
    V    : ndarray (n x n) -- vectores propios normalizados (columnas)
    er_k : float           -- error alcanzado en la ultima iteracion
    k    : int             -- iteraciones realizadas
    conv : int             -- 1 si convergio, 0 si no
    """
    n = A.shape[0]

    Ak = A.copy().astype(float)   # A_0 = A
    V = np.eye(n)                 # V_0 = I

    lam_ant = np.diag(Ak).copy()  # diag(A_0), para el primer error
    er_k = tol + 1                # fuerza a entrar al ciclo
    k = 0
    conv = 0

    while (er_k > tol) and (k < iterMax):
        # --- factorizacion QR de la iteracion (biblioteca) ---
        Q, R = np.linalg.qr(Ak)

        # --- A_{k+1} = R_k * Q_k ---
        Ak = R @ Q

        # --- se acumulan los vectores propios: V_{k+1} = V_k * Q_k ---
        V = V @ Q

        # --- error = cambio de la diagonal respecto a la iteracion anterior ---
        lam_act = np.diag(Ak).copy()
        er_k = np.linalg.norm(lam_act - lam_ant)
        lam_ant = lam_act

        k = k + 1

    if er_k <= tol:
        conv = 1

    lam, V = _ordenar_y_normalizar(Ak, V)
    return lam, V, er_k, k, conv


# =====================================================================
# 2. Metodo de Jacobi por rotaciones (metodo investigado)
# =====================================================================

def metodo_jacobi(A, iterMax, tol):
    """
    Calcula los valores y vectores propios de una matriz simetrica A
    mediante el Metodo de Jacobi por rotaciones. Este metodo NO fue
    visto en clase; se implementa aqui de forma completamente manual
    (sin ninguna llamada a funciones de biblioteca para el algebra
    lineal del metodo en si), tal como lo pide el enunciado para los
    metodos investigados del Bloque 3.

    Idea del metodo:
        En cada iteracion se localiza el elemento fuera de la
        diagonal de mayor valor absoluto, A(p,q) (pivoteo clasico),
        y se aplica una rotacion ortogonal J(p,q,theta) que lo anula:

            A_{k+1} = J^T * A_k * J

        El angulo theta se obtiene de forma numericamente estable
        mediante:

            tau = (A(q,q) - A(p,p)) / (2*A(p,q))
            t   = sign(tau) / (|tau| + sqrt(1 + tau^2))
            c   = 1 / sqrt(1 + t^2)
            s   = t * c

        (con sign(0) = 1 cuando tau = 0). Este valor de t es la raiz
        de menor magnitud de t^2 + 2*t*tau - 1 = 0, lo que evita la
        cancelacion catastrofica que ocurriria al calcular
        -tau + sqrt(1+tau^2) directamente cuando |tau| es grande.

        Los vectores propios se acumulan como V_{k+1} = V_k * J.

    Criterio de parada:
        er_k = || diag(A_k) - diag(A_{k-1}) ||_2 < tol
        (el mismo criterio usado en metodo_qr, para mantener
        consistencia entre ambos metodos del bloque).

    Parametros
    ----------
    A : ndarray (n x n)
        Matriz simetrica cuyos valores y vectores propios se buscan.
    iterMax : int
        Numero maximo de iteraciones permitidas.
    tol : float
        Tolerancia para el criterio de parada.

    Retorna
    -------
    lam  : ndarray (n x 1) -- valores propios, de mayor a menor
    V    : ndarray (n x n) -- vectores propios normalizados (columnas)
    er_k : float           -- error alcanzado en la ultima iteracion
    k    : int             -- iteraciones realizadas
    conv : int             -- 1 si convergio, 0 si no
    """
    n = A.shape[0]

    Ak = A.copy().astype(float)   # A_0 = A
    V = np.eye(n)                 # V_0 = I

    lam_ant = np.diag(Ak).copy()  # diag(A_0), para el primer error
    er_k = tol + 1                # fuerza a entrar al ciclo
    k = 0
    conv = 0

    while (er_k > tol) and (k < iterMax):

        # --- 1. pivoteo clasico: localizar |A(p,q)| maximo fuera de la diagonal ---
        # (la diagonal se excluye con -inf, para que nunca sea elegida
        # como pivote, incluso cuando A_k ya es diagonal)
        off = np.abs(Ak).copy()
        np.fill_diagonal(off, -np.inf)
        p, q = np.unravel_index(np.argmax(off), off.shape)
        if p > q:
            p, q = q, p

        apq = Ak[p, q]

        if apq == 0:
            # A_k ya es diagonal: no hay nada que rotar, se alcanzo
            # la convergencia exacta antes de agotar iterMax
            er_k = 0.0
            conv = 1
            break

        # --- 2. calculo estable del angulo de rotacion (c, s) ---
        tau = (Ak[q, q] - Ak[p, p]) / (2 * apq)
        if tau >= 0:
            t = 1.0 / (tau + np.sqrt(1 + tau**2))
        else:
            t = -1.0 / (-tau + np.sqrt(1 + tau**2))
        c = 1.0 / np.sqrt(1 + t**2)
        s = t * c

        # --- 3. construir J(p,q,theta) y aplicar la rotacion A_{k+1} = J^T*A_k*J ---
        J = np.eye(n)
        J[p, p] = c
        J[q, q] = c
        J[p, q] = s
        J[q, p] = -s

        Ak = J.T @ Ak @ J
        V = V @ J   # se acumulan los vectores propios

        # --- 4. error = cambio de la diagonal respecto a la iteracion anterior ---
        lam_act = np.diag(Ak).copy()
        er_k = np.linalg.norm(lam_act - lam_ant)
        lam_ant = lam_act

        k = k + 1

    if er_k <= tol:
        conv = 1

    lam, V = _ordenar_y_normalizar(Ak, V)
    return lam, V, er_k, k, conv