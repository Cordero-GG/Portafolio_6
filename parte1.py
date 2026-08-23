import sympy as sp

def evaluar_f(f_str, val):
    """Convierte la cadena de texto a expresión matemática y evalúa en un número."""
    x = sp.Symbol('x')
    expr = sp.sympify(f_str)
    return float(expr.subs(x, val))

def derivar_f(f_str):
    """Devuelve la expresión simbólica de la primera derivada."""
    x = sp.Symbol('x')
    expr = sp.sympify(f_str)
    return sp.diff(expr, x)

# 1. BISECCIÓN
def biseccion(f_str, a, b, iterMax, tol):
    k = 0
    conv = 0
    fa = evaluar_f(f_str, a)
    fb = evaluar_f(f_str, b)
    
    if fa * fb >= 0:
        return None, None, 0, 0  # No hay cambio de signo
        
    x_k = a
    er_k = float('inf')
    
    while k < iterMax:
        k += 1
        x_k_prev = x_k
        x_k = (a + b) / 2.0
        fx = evaluar_f(f_str, x_k)
        
        if k > 1:
            er_k = abs(x_k - x_k_prev)
            if er_k < tol or abs(fx) < tol:
                conv = 1
                break
                
        if fa * fx < 0:
            b = x_k
            fb = fx
        else:
            a = x_k
            fa = fx
            
    return x_k, er_k, k, conv

# 2. FALSA POSICIÓN
def falsa_posicion(f_str, a, b, iterMax, tol):
    k = 0
    conv = 0
    fa = evaluar_f(f_str, a)
    fb = evaluar_f(f_str, b)
    
    if fa * fb >= 0:
        return None, None, 0, 0
        
    x_k = a
    er_k = float('inf')
    
    while k < iterMax:
        k += 1
        x_k_prev = x_k
        x_k = b - (fb * (b - a)) / (fb - fa)
        fx = evaluar_f(f_str, x_k)
        
        if k > 1:
            er_k = abs(x_k - x_k_prev)
            if er_k < tol or abs(fx) < tol:
                conv = 1
                break
                
        if fa * fx < 0:
            b = x_k
            fb = fx
        else:
            a = x_k
            fa = fx
            
    return x_k, er_k, k, conv

# 3. NEWTON-RAPHSON
def newton_raphson(f_str, x0, iterMax, tol):
    k = 0
    conv = 0
    df_expr = derivar_f(f_str)
    x = sp.Symbol('x')
    x_k = float(x0)
    er_k = float('inf')
    
    while k < iterMax:
        k += 1
        fx = evaluar_f(f_str, x_k)
        dfx = float(df_expr.subs(x, x_k))
        
        if dfx == 0:
            break
            
        x_k_next = x_k - (fx / dfx)
        er_k = abs(x_k_next - x_k)
        x_k = x_k_next
        
        if er_k < tol:
            conv = 1
            break
            
    return x_k, er_k, k, conv

# 4. SECANTE
def secante(f_str, x0, x1, iterMax, tol):
    k = 0
    conv = 0
    er_k = float('inf')
    
    while k < iterMax:
        k += 1
        f_x0 = evaluar_f(f_str, x0)
        f_x1 = evaluar_f(f_str, x1)
        
        if f_x1 - f_x0 == 0:
            break
            
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        er_k = abs(x2 - x1)
        x0, x1 = x1, x2
        
        if er_k < tol:
            conv = 1
            break
            
    return x1, er_k, k, conv

# 5. STEFFENSEN
def steffensen(f_str, x0, iterMax, tol):
    k = 0
    conv = 0
    x_k = float(x0)
    er_k = float('inf')
    
    while k < iterMax:
        k += 1
        fx = evaluar_f(f_str, x_k)
        f_x_plus_fx = evaluar_f(f_str, x_k + fx)
        
        denom = f_x_plus_fx - fx
        if denom == 0:
            break
            
        x_k_next = x_k - (fx**2) / denom
        er_k = abs(x_k_next - x_k)
        x_k = x_k_next
        
        if er_k < tol:
            conv = 1
            break
            
    return x_k, er_k, k, conv

# 6. MÜLLER
def muller(f_str, x0, x1, x2, iterMax, tol):
    k = 0
    conv = 0
    er_k = float('inf')
    
    while k < iterMax:
        k += 1
        f0 = evaluar_f(f_str, x0)
        f1 = evaluar_f(f_str, x1)
        f2 = evaluar_f(f_str, x2)
        
        h0 = x1 - x0
        h1 = x2 - x1
        d0 = (f1 - f0) / h0
        d1 = (f2 - f1) / h1
        
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = f2
        
        disc = (b**2 - 4*a*c)**0.5
        denom = b + disc if abs(b + disc) > abs(b - disc) else b - disc
        
        if denom == 0:
            break
            
        dx = -2 * c / denom
        x3 = x2 + dx
        er_k = abs(dx)
        
        x0, x1, x2 = x1, x2, x3
        
        if er_k < tol:
            conv = 1
            break
            
    return x2, er_k, k, conv