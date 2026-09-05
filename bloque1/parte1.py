import sympy as sp

def evaluar_f(funcion_str, val):
    """Convierte la cadena de texto a expresión matemática y evalúa en un número."""
    x = sp.Symbol('x')
    expr = sp.sympify(funcion_str)
    return float(expr.subs(x, val))

def derivar_funciones(funcion_str):
    """Primera derivada."""
    x = sp.Symbol('x')
    expr = sp.sympify(funcion_str)
    return sp.diff(expr, x)

def crear_funcion_numerica(funcion_str):
    """Convierte la cadena de texto en una función numérica rápida."""
    x = sp.Symbol('x')
    expr = sp.sympify(funcion_str)
    return sp.lambdify(x, expr, "numpy")

def crear_derivada_numerica(funcion_str):
    """Deriva simbólicamente y convierte el resultado en función numérica rápida."""
    x = sp.Symbol('x')
    expr = sp.sympify(funcion_str)
    derivada = sp.diff(expr, x)
    return sp.lambdify(x, derivada, "numpy")

#Newton-Raphson
def newton_raphson(funcion_str, x0, maxiIteraciones, tolerancia):
    f = crear_funcion_numerica(funcion_str)
    fp = crear_derivada_numerica(funcion_str)
    k = 0
    conv = 0
    xk = float(x0) #estimacion actual
    erk = float('inf')

    while k <maxiIteraciones:
        k += 1
        funcionEvaluada = f(xk)
        derivadaEvaluada = fp(xk)
        if derivadaEvaluada == 0:
            return xk, erk, k, conv
        
        xksiguiente = xk - (funcionEvaluada/derivadaEvaluada)
        erk = abs(xksiguiente - xk)
        xk = xksiguiente

        if erk < tolerancia:
            conv = 1
            break
    return xk, erk, k, conv

# 2. Secante
def secante(funcion_str, x0, x1, maxiIteraciones, tolerancia):
    f = crear_funcion_numerica(funcion_str)
    k = 0
    conv = 0
    x0 = float(x0)
    x1 = float(x1)
    erk = float('inf')

    while k < maxiIteraciones:
        k += 1
        f_x0 = f(x0)
        f_x1 = f(x1)

        # Evitar división por cero
        denominador = f_x1 - f_x0
        if denominador == 0:
            return x1, erk, k, conv

        # Fórmula de la Secante
        x2 = x1 - f_x1 * (x1 - x0) / denominador
        erk = abs(x2 - x1)

        x0 = x1
        x1 = x2

        if erk < tolerancia:
            conv = 1
            break

    return x1, erk, k, conv


# 3. Steffensen
def steffensen(funcion_str, x0, maxiIteraciones, tolerancia):
    f = crear_funcion_numerica(funcion_str)
    k = 0
    conv = 0
    xk = float(x0)
    erk = float('inf')

    while k < maxiIteraciones:
        k += 1
        funcionEvaluada = f(xk)
        
        # Se evalúa f en el punto desplazado (xk + f(xk))
        funcionDesplazada = f(xk + funcionEvaluada)
        
        denominador = funcionDesplazada - funcionEvaluada
        if denominador == 0:
            return xk, erk, k, conv
            
        xksiguiente = xk - (funcionEvaluada**2) / denominador
        erk = abs(xksiguiente - xk)
        xk = xksiguiente
        
        if erk < tolerancia:
            conv = 1
            break

    return xk, erk, k, conv


# 4. Müller
def muller(funcion_str, x0, x1, x2, maxiIteraciones, tolerancia):
    f = crear_funcion_numerica(funcion_str)
    k = 0
    conv = 0
    x0, x1, x2 = float(x0), float(x1), float(x2)
    erk = float('inf')

    while k < maxiIteraciones:
        k += 1
        f0, f1, f2 = f(x0), f(x1), f(x2)
        c = f2
        denom_comun = (x0 - x1) * (x0 - x2) * (x1 - x2)
        
        if denom_comun == 0:
            return x2, erk, k, conv

        b = ((x0 - x2)**2 * (f1 - f2) - (x1 - x2)**2 * (f0 - f2)) / denom_comun
        a = ((x1 - x2) * (f0 - f2) - (x0 - x2) * (f1 - f2)) / denom_comun

        radicando = complex(b**2 - 4 * a * c)
        raiz_disc = radicando ** 0.5
        sgn_b = 1 if b >= 0 else -1
        denominador = b + sgn_b * raiz_disc
        denominador = denominador.real if isinstance(denominador, complex) else denominador

        if denominador == 0:
            return x2, erk, k, conv

        r = x2 - 2 * c / denominador
        erk = abs(r - x2)
        # Ordeno los puntos viejos para ver los que están más cerca de r
        puntos_viejos = sorted([x0, x1, x2], key=lambda x: abs(x - r))
        # Los 2 más cercanos y x2 pasa a ser la nueva raíz
        x0, x1 = puntos_viejos[0], puntos_viejos[1]
        x2 = r
        if erk < tolerancia:
            conv = 1
            break

    return x2, erk, k, conv


# 5. Bisección
def biseccion(funcion_str, a, b, maxiIteraciones, tolerancia):
    f = crear_funcion_numerica(funcion_str)
    k = 0
    conv = 0
    a = float(a)
    b = float(b)
    fa = f(a)
    fb = f(b)

    #Si la raíz está en los extremos
    if fa == 0:
        return a, 0.0, 0, 1
    if fb == 0:
        return b, 0.0, 0, 1

    # Si no hay cambio de signo no se garantiza la convergencia
    if fa * fb > 0:
        return None, None, None, 0

    #Inicia el ciclo
    xk = a
    erk = float('inf')

    while k < maxiIteraciones:
        k += 1
        xk = (a + b) / 2.0
        funcionEvaluada = f(xk)
        erk = abs(funcionEvaluada)
        if erk < tolerancia:
            conv = 1
            break
        #nuevo intervalo
        if fa * funcionEvaluada < 0:
            b = xk
            fb = funcionEvaluada 
        else:
            a = xk
            fa = funcionEvaluada

    return xk, erk, k, conv


# 6. Falsa Posición
def falsa_posicion(funcion_str, a, b, maxiIteraciones, tolerancia):
    f = crear_funcion_numerica(funcion_str)
    k = 0
    conv = 0
    a = float(a)
    b = float(b)
    fa = f(a)
    fb = f(b)

    #Si la raíz exacta está en los bordes
    if fa == 0:
        return a, 0.0, 0, 1
    if fb == 0:
        return b, 0.0, 0, 1
    #Si no hay cambio de signo
    if fa * fb > 0:
        return None, None, None, 0
    #Ciclo iterativo
    xk = a
    erk = float('inf')

    while k < maxiIteraciones:
        k += 1
        denominador = fb - fa
        if denominador == 0:
            return xk, erk, k, conv
            
        # Fórmula de la secante para nuevo punto
        xk = b - (fb * (b - a)) / denominador
        funcionEvaluada = f(xk)
        erk = abs(funcionEvaluada)
        
        if erk < tolerancia:
            conv = 1
            break
        #nuevo subintervalo
        if fa * funcionEvaluada < 0:
            b = xk
            fb = funcionEvaluada
        else:
            a = xk
            fa = funcionEvaluada

    return xk, erk, k, conv  