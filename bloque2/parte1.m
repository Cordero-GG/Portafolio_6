function M = parte1()
  M.eliminacion_gaussiana = @eliminacion_gaussiana;
  M.factorizacion_LU      = @factorizacion_LU;
  M.cholesky              = @cholesky;
  M.factorizacion_QR      = @factorizacion_QR;
  M.metodo_thomas         = @metodo_thomas;
  M.jacobi                = @jacobi;
  M.gauss_seidel          = @gauss_seidel;
  M.gradiente_conjugado   = @gradiente_conjugado;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Eliminacion gaussiana
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = eliminacion_gaussiana(A, b)
  % resuelve Ax=b transformando (A|b) en un sistema equivalente (As|bs) triangular superior y luego resolviendo As*x=bs por sustitucion hacia atras

  [As, bs] = triangularizar_gauss(A, b);
  x = sustitucion_atras(As, bs);
end


function [As, bs] = triangularizar_gauss(A, b)
  % aplica eliminacion gaussiana sin pivoteo. Para cada columna k, elimina los elementos debajo del pivote A(k,k) usando el multiplicador m = A(i,k)/A(k,k) en cada fila i>k.

  n = size(A, 1);
  As = A;
  bs = b;

  for k = 1:n-1 % columna que se está anulando
    for i = k+1:n % filas debajo del pivote
      m = As(i,k) / As(k,k); % multiplicador
      As(i,k:n) = As(i,k:n) - m*As(k,k:n); % misma operacion fila a fila
      bs(i) = bs(i) - m*bs(k); % misma operacion sobre b
    end
  end
end


function x = sustitucion_atras(A, b)
  % resuelve Ax=b para A triangular superior, recorriendo las filas de abajo hacia arriba (i=n,...,1).

  n = size(A, 1);
  x = zeros(n, 1);

  for i = n:-1:1
    suma = A(i,i+1:n) * x(i+1:n); % terminos ya conocidos
    x(i) = (b(i) - suma) / A(i,i);
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Factorizacion LU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = factorizacion_LU(A, b)
  % Resuelve Ax=b mediante A=LU (L triangular inferior con diagonal de 1s, U triangular superior)

  [L, U] = fact_LU(A);
  y = sustitucion_adelante(L, b);
  x = sustitucion_atras(U, y);
end


function [L, U] = fact_LU(A)
  % calcula la factorizacion A=LU sin pivoteo. U se obtiene igual que en eliminacion gaussiana; L guarda los multiplicadores usados en cada paso.

  n = size(A, 1);
  U = A;
  L = eye(n); % L inicia como identidad (diagonal de 1s)

  for k = 1:n-1 % columna que se está anulando
    for i = k+1:n % filas debajo del pivote
      m = U(i,k) / U(k,k); % multiplicador
      L(i,k) = m; % se guarda debajo de la diagonal de L
      U(i,k:n) = U(i,k:n) - m*U(k,k:n);
    end
  end
end


function y = sustitucion_adelante(L, c)
  % resuelve L*y=c para L triangular inferior, recorriendo las filas de arriba hacia abajo (i=1,...,n).

  n = size(L, 1);
  y = zeros(n, 1);

  for i = 1:n
    suma = L(i,1:i-1) * y(1:i-1); % terminos ya conocidos
    y(i) = (c(i) - suma) / L(i,i);
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Factorizacion de Cholesky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = cholesky(A, b)
  % resuelve Ax=b cuando A es simetrica definida positiva (SPD), mediante A=L*L'. Se resuelve Ly=b hacia adelante y L'x=y hacia atras

  L = fact_cholesky(A);
  y = sustitucion_adelante(L, b);
  x = sustitucion_atras(L', y);
end


function L = fact_cholesky(A)
  % calcula la factorizacion de Cholesky A=L*L'. Primero se calcula la columna 1; luego, para cada columna i=2,...,n, se calcula el elemento diagonal y despues los elementos debajo de la diagonal.

  n = size(A, 1);
  L = zeros(n, n);

  L(1,1) = sqrt(A(1,1)); % columna 1: elemento diagonal
  for j = 2:n
    L(j,1) = A(j,1) / L(1,1); % columna 1: elementos debajo de la diagonal
  end

  for i = 2:n
    suma1 = L(i,1:i-1) * L(i,1:i-1)'; % suma de los cuadrados de la fila i
    L(i,i) = sqrt(A(i,i) - suma1); % elemento diagonal de la columna i

    for j = i+1:n
      suma2 = L(j,1:i-1) * L(i,1:i-1)';
      L(j,i) = (A(j,i) - suma2) / L(i,i); % elementos debajo de la diagonal
    end
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. Factorizacion QR (Gram-Schmidt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = factorizacion_QR(A, b)
  % resuelve Ax=b mediante A=QR (Q ortogonal, R triangular superior). Como Q'Q=I, el sistema se reduce a Rx=Q'b, resuelto por sustitucion hacia atras

  [Q, R] = fact_QR(A);
  c = Q' * b;
  x = sustitucion_atras(R, c);
end


function [Q, R] = fact_QR(A)
  % calcula la factorizacion A=QR mediante ortonormalizacion de Gram-Schmidt: a cada columna a_k se le resta su proyeccion sobre los q_j ya calculados (j<k), y se normaliza el resultado

  n = size(A, 1);
  Q = zeros(n, n);

  for k = 1:n
    uk = A(:,k);
    for j = 1:k-1
      uk = uk - (A(:,k)' * Q(:,j)) * Q(:,j); % restar proyeccion sobre q_j
    end
    Q(:,k) = uk / norm(uk); % normalizar
  end

  R = Q' * A; % R queda triangular superior
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. Metodo de Thomas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = metodo_thomas(A, b)
  % resuelve Ax=b cuando A es tridiagonal, en O(n) operaciones. Fase 1: barrido hacia adelante (calcula p y q). Fase 2: sustitucion hacia atras (calcula x)

  n = size(A, 1);

  d = diag(A); % diagonal principal
  cs = diag(A, 1); % superdiagonal
  as = diag(A, -1); % subdiagonal
  as = [0; as]; % se antepone un 0 para indexar igual que d y cs

  p = zeros(n-1, 1);
  q = zeros(n, 1);

  p(1) = cs(1) / d(1);
  q(1) = b(1) / d(1);

  for i = 2:n-1
    aux = d(i) - p(i-1)*as(i);
    p(i) = cs(i) / aux;
    q(i) = (b(i) - q(i-1)*as(i)) / aux;
  end

  q(n) = (b(n) - q(n-1)*as(n)) / (d(n) - p(n-1)*as(n));

  x = zeros(n, 1);
  x(n) = q(n);

  for i = n-1:-1:1
    x(i) = q(i) - p(i)*x(i+1); % sustitucion hacia atras
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6. Metodo de Jacobi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xk, erk, k, conv] = jacobi(A, b, x0, tol, iterMax)
  % resuelve Ax=b de forma iterativa usando A=L+D+U: x^(k+1) = D^-1[b-(L+U)x^(k)]. D^-1 nunca se calcula con un algoritmo general: se guarda el vector de reciprocos (dinv) y se aplica con producto Hadamard. Criterio de parada: ||Ax^(k)-b||_2 < tol

  n = size(A, 1);

  dinv = zeros(n, 1);
  for i = 1:n
    dinv(i) = 1 / A(i,i); % reciprocos de la diagonal
  end

  D = diag(diag(A));
  R = A - D; % R = L + U

  xk = x0;
  erk = norm(A*xk - b); % error = residuo
  k = 0;
  conv = 0;

  while (erk > tol) && (k < iterMax)
    xk = dinv .* (b - R*xk); % D^-1*(b-R*xk), componente a componente
    erk = norm(A*xk - b);
    k = k + 1;
  end

  if erk <= tol
    conv = 1;
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7. Metodo de Gauss-Seidel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xk, erk, k, conv] = gauss_seidel(A, b, x0, tol, iterMax)
  % resuelve Ax=b de forma iterativa usando A=L+D+U: (L+D)x^(k+1) = b-Ux^(k). NO se calcula (L+D)^-1: en cada iteracion se resuelve el sistema triangular inferior por sustitucion hacia adelante. Criterio de parada: ||Ax^(k)-b||_2 < tol

  L = tril(A, -1);
  D = diag(diag(A));
  U = triu(A, 1);
  M = L + D; % triangular inferior

  xk = x0;
  erk = norm(A*xk - b); % error = residuo
  k = 0;
  conv = 0;

  while (erk > tol) && (k < iterMax)
    c = b - U*xk;
    xk = sustitucion_adelante(M, c); % resolver M*x_nuevo = c
    erk = norm(A*xk - b);
    k = k + 1;
  end

  if erk <= tol
    conv = 1;
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 8. Gradiente Conjugado
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xk, erk, k, conv] = gradiente_conjugado(A, b, x0, tol, iterMax)
  % resuelve Ax=b (A simetrica definida positiva) construyendo direcciones de busqueda pk conjugadas respecto de A. Criterio de parada: ||Ax^(k)-b||_2 < tol

  xk = x0;
  rk = b - A*xk; % residuo inicial
  pk = rk; % primera direccion de busqueda = residuo inicial
  erk = norm(rk);
  k = 0;
  conv = 0;

  % se agrega la condicion rk'*rk > 0: si el residuo recursivo llega a ser
  % exactamente cero, el metodo ya alcanzo la solucion y no existe una nueva
  % direccion de busqueda (alpha y beta quedarian indefinidos como 0/0)
  while (erk > tol) && (k < iterMax) && (rk' * rk > 0)
    Apk = A*pk;
    alpha = (rk' * rk) / (pk' * Apk); % tamano de paso optimo

    xk = xk + alpha*pk; % nueva aproximacion
    rk_nuevo = rk - alpha*Apk; % nuevo residuo

    beta = (rk_nuevo' * rk_nuevo) / (rk' * rk); % correccion de direccion
    pk = rk_nuevo + beta*pk; % nueva direccion conjugada
    rk = rk_nuevo;

    erk = norm(A*xk - b);
    k = k + 1;
  end

  if erk <= tol
    conv = 1;
  end
end
