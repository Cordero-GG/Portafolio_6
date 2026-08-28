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
    x0 = float(x0)
    x1 = float(x1)
    x2 = float(x2)
    erk = float('inf')

    while k < maxiIteraciones:
        k += 1
        f0 = f(x0)
        f1 = f(x1)
        f2 = f(x2)

        h0 = x1 - x0
        h1 = x2 - x1
        
        if h0 == 0 or h1 == 0:
            return x2, erk, k, conv

        d0 = (f1 - f0) / h0
        d1 = (f2 - f1) / h1

        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = f2

        discriminante = (b**2 - 4 * a * c)**0.5
        
        # Signo que maximice el denominador para evitar divisiones inestables
        if abs(b + discriminante) > abs(b - discriminante):
            denominador = b + discriminante
        else:
            denominador = b - discriminante

        if denominador == 0:
            return x2, erk, k, conv

        dx = -2 * c / denominador
        x3 = x2 + dx
        erk = abs(dx)

        x0 = x1
        x1 = x2
        x2 = x3

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

    if fa * fb >= 0:
        return None, None, 0, 0  

    xk = a
    erk = float('inf')

    while k < maxiIteraciones:
        k += 1
        xksiguiente = (a + b) / 2.0
        funcionEvaluada = f(xksiguiente)

        if k > 1:
            erk = abs(xksiguiente - xk)
            if erk < tolerancia or abs(funcionEvaluada) < tolerancia:
                conv = 1
                xk = xksiguiente
                break

        xk = xksiguiente

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

    if fa * fb >= 0:
        return None, None, 0, 0  
    xk = a
    erk = float('inf')

    while k < maxiIteraciones:
        k += 1
        denominador = fb - fa
        if denominador == 0:
            return xk, erk, k, conv

        xksiguiente = b - (fb * (b - a)) / denominador
        funcionEvaluada = f(xksiguiente)

        if k > 1:
            erk = abs(xksiguiente - xk)
            if erk < tolerancia or abs(funcionEvaluada) < tolerancia:
                conv = 1
                xk = xksiguiente
                break

        xk = xksiguiente

        if fa * funcionEvaluada < 0:
            b = xk
            fb = funcionEvaluada
        else:
            a = xk
            fa = funcionEvaluada

    return xk, erk, k, conv    