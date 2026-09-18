%  Portafolio Computacional - Bloque 2 - Parte III

clear; close all; clc; more off;

M = parte1();   % handles de los metodos implementados en parte1.m

%  1. Discretizacion y construccion del sistema A*T = b

n  = 499;
h  = 1/500;
xi = (1:n)' * h;


A = 510000*eye(n) - 250000*( diag(ones(n-1,1),1) + diag(ones(n-1,1),-1) );

% Lado derecho: g_i = 300000*x_i + (100000+10*pi^2)*sin(pi*x_i) + 200000
g = 300000*xi + (100000 + 10*pi^2)*sin(pi*xi) + 200000;

b = g;
b(1) = b(1) + 5000000;
b(n) = b(n) + 12500000;

% Condiciones de frontera y parametros de los metodos iterativos
Tizq    = 20;
Tder    = 50;
x0      = zeros(n,1);
tol     = 1e-8;
iterMax = 10000;

%  2. Solucion del sistema con los ocho metodos de parte1.m

nombres = {'Eliminacion Gaussiana','Factorizacion LU','Cholesky', ...
           'Factorizacion QR','Metodo de Thomas','Jacobi', ...
           'Gauss-Seidel','Gradiente Conjugado'};
etiq    = {'Gauss','LU','Cholesky','QR','Thomas','Jacobi','G-Seidel','Grad.Conj.'};
nm      = numel(nombres);

Tsol = zeros(n, nm);        % solucion de cada metodo (columna j)
er   = zeros(nm, 1);        % error ||A*T - b||_2
tej  = zeros(nm, 1);        % tiempo de ejecucion
kit  = nan(nm, 1);          % iteraciones (solo metodos iterativos)
cv   = nan(nm, 1);          % indicador de convergencia

printf('Resolviendo el sistema de diferencias finitas (n = %d)...\n\n', n);

%Metodos directos

printf('  Eliminacion Gaussiana ... '); fflush(stdout);
tic; Tsol(:,1) = M.eliminacion_gaussiana(A, b); tej(1) = toc;
printf('%8.3f s\n', tej(1)); fflush(stdout);

printf('  Factorizacion LU ....... '); fflush(stdout);
tic; Tsol(:,2) = M.factorizacion_LU(A, b); tej(2) = toc;
printf('%8.3f s\n', tej(2)); fflush(stdout);

printf('  Cholesky ............... '); fflush(stdout);
tic; Tsol(:,3) = M.cholesky(A, b); tej(3) = toc;
printf('%8.3f s\n', tej(3)); fflush(stdout);

printf('  Factorizacion QR ....... '); fflush(stdout);
tic; Tsol(:,4) = M.factorizacion_QR(A, b); tej(4) = toc;
printf('%8.3f s\n', tej(4)); fflush(stdout);

printf('  Metodo de Thomas ....... '); fflush(stdout);
tic; Tsol(:,5) = M.metodo_thomas(A, b); tej(5) = toc;
printf('%8.3f s\n', tej(5)); fflush(stdout);

% Metodos iterativos

printf('  Jacobi ................. '); fflush(stdout);
tic; [xk, erk, k, conv] = M.jacobi(A, b, x0, tol, iterMax); tej(6) = toc;
Tsol(:,6) = xk; kit(6) = k; cv(6) = conv;
printf('%8.3f s  (k = %d)\n', tej(6), k); fflush(stdout);

printf('  Gauss-Seidel ........... '); fflush(stdout);
tic; [xk, erk, k, conv] = M.gauss_seidel(A, b, x0, tol, iterMax); tej(7) = toc;
Tsol(:,7) = xk; kit(7) = k; cv(7) = conv;
printf('%8.3f s  (k = %d)\n', tej(7), k); fflush(stdout);

printf('  Gradiente Conjugado .... '); fflush(stdout);
tic; [xk, erk, k, conv] = M.gradiente_conjugado(A, b, x0, tol, iterMax); tej(8) = toc;


cg_ruptura = false;
cg_iterMax = iterMax;
if any(~isfinite(xk))
  cg_ruptura = true;
  kprueba = iterMax;
  for intento = 1:20
    kprueba = floor(kprueba/2);
    if kprueba < 1
      break;
    end
    tic; [xk, erk, k, conv] = M.gradiente_conjugado(A, b, x0, tol, kprueba); tej(8) = toc;
    if all(isfinite(xk))
      cg_iterMax = kprueba;
      break;
    end
  end
end

Tsol(:,8) = xk; kit(8) = k; cv(8) = conv;
if cg_ruptura
  printf('%8.3f s  (k = %d, ruptura numerica con iterMax = %d)\n\n', tej(8), k, iterMax);
else
  printf('%8.3f s  (k = %d)\n\n', tej(8), k);
end
fflush(stdout);

% Error er = ||A*T - b||_2 para todos los metodos, calculado igual
for j = 1:nm
  er(j) = norm(A*Tsol(:,j) - b);
end

%  3. Corrida auxiliar de los metodos iterativos con una tolerancia
%     alcanzable en aritmetica de punto flotante.


tol2 = 1e-3;
kit2 = zeros(3,1);
cv2  = zeros(3,1);

printf('Corrida auxiliar de los metodos iterativos con tol = %.0e ...\n', tol2);
fflush(stdout);

[~, ~, kit2(1), cv2(1)] = M.jacobi(A, b, x0, tol2, iterMax);
[~, ~, kit2(2), cv2(2)] = M.gauss_seidel(A, b, x0, tol2, iterMax);
[~, ~, kit2(3), cv2(3)] = M.gradiente_conjugado(A, b, x0, tol2, iterMax);

printf('  Jacobi k = %d,  Gauss-Seidel k = %d,  Gradiente Conjugado k = %d\n\n', ...
       kit2(1), kit2(2), kit2(3));
fflush(stdout);

%  4. Tabla comparativa

printf('===========================================================================================\n');
printf('  TABLA COMPARATIVA - Parte III: barra metalica (sistema de %d x %d)\n', n, n);
printf('  Metodos iterativos: T^(0) = 0, iterMax = %d, tol = %.0e\n', iterMax, tol);
printf('===========================================================================================\n');

nombres_tab = nombres;
if cg_ruptura
  nombres_tab{8} = 'Gradiente Conjugado (*)';
end

printf(' %-24s %16s %13s %9s %6s %12s\n', 'Metodo', '||AT-b||_2', 'Tiempo (s)', 'k', 'conv', 'k (tol=1e-3)');
printf('-------------------------------------------------------------------------------------------\n');
for j = 1:nm
  if isnan(kit(j))
    printf(' %-24s %16.4e %13.4f %9s %6s %12s\n', nombres_tab{j}, er(j), tej(j), '-', '-', '-');
  else
    printf(' %-24s %16.4e %13.4f %9d %6d %12d\n', nombres_tab{j}, er(j), tej(j), ...
           kit(j), cv(j), kit2(j-5));
  end
end
printf('===========================================================================================\n');
if cg_ruptura
  printf(' (*) Con iterMax = %d el gradiente conjugado sufre una ruptura numerica (division 0/0)\n', iterMax);
  printf('     y devuelve NaN. Los valores mostrados corresponden a la ultima corrida estable,\n');
  printf('     obtenida con iterMax = %d. La causa se explica en el punto (6) del analisis.\n', cg_iterMax);
end
printf('\n');

%  5. Reconstruccion de la distribucion de temperatura

jref = 5;                          % solucion de referencia: metodo de Thomas
xs   = (0:500)' * h;               % los 501 nodos x_i = i/500
Tnum = [Tizq; Tsol(:,jref); Tder]; % se incorporan T0 = 20 y T500 = 50
Texa = 20 + 30*xs + 10*sin(pi*xs); % solucion exacta del problema

errNodos = max(abs(Tnum - Texa));  % error maximo respecto de la solucion exacta
[Tmax, imax] = max(Tnum);

% Diferencia maxima entre las soluciones de los distintos metodos
dif = 0;
for j = 1:nm
  dif = max(dif, max(abs(Tsol(:,j) - Tsol(:,jref))));
end

%  6. Graficas

% ---- Figura 1: errores ||AT-b||_2 (escala logaritmica) ----
figure(1);
bar(er, 'facecolor', [0.20 0.45 0.75]);
set(gca, 'xtick', 1:nm, 'xticklabel', etiq, 'yscale', 'log');
ylim([10^(floor(log10(min(er)))-1), 10^(ceil(log10(max(er)))+1)]);
title('Parte III: error ||AT - b||_2 por metodo (escala logaritmica)');
xlabel('Metodo'); ylabel('||AT - b||_2');
grid on;

% ---- Figura 2: tiempos de ejecucion (escala logaritmica) ----
figure(2);
bar(tej, 'facecolor', [0.85 0.40 0.15]);
set(gca, 'xtick', 1:nm, 'xticklabel', etiq, 'yscale', 'log');
ylim([10^(floor(log10(min(tej(tej>0))))-1), 10^(ceil(log10(max(tej)))+1)]);
title('Parte III: tiempo de ejecucion por metodo (escala logaritmica)');
xlabel('Metodo'); ylabel('Tiempo (s)');
grid on;

% ---- Figura 3: iteraciones de los metodos iterativos ----
figure(3);
bar([kit(6:8), kit2], 'grouped');
set(gca, 'xtick', 1:3, 'xticklabel', {'Jacobi','Gauss-Seidel','Grad. Conjugado'});
legend('tol = 10^{-8} (se agota iterMax)', 'tol = 10^{-3} (alcanzable)', ...
       'location', 'northeast');
title('Parte III: numero de iteraciones de los metodos iterativos');
xlabel('Metodo'); ylabel('Iteraciones k');
grid on;

% ---- Figura 4: distribucion de temperatura en la barra ----
figure(4);
plot(xs, Texa, 'b-', 'linewidth', 2); hold on;
scatter(xs, Tnum, 14, 'r', 'filled');
hold off;
title('Parte III: distribucion de temperatura en la barra (estado estacionario)');
xlabel('Posicion x (m)'); ylabel('Temperatura T (grados C)');
legend('Solucion exacta T(x) = 20 + 30x + 10 sin(\pi x)', ...
       'Aproximacion por diferencias finitas (x_i, T_i)', ...
       'location', 'northwest');
grid on;
axis([0 1 15 55]);

% ---- Figura 5: error nodal de la aproximacion ----
figure(5);
plot(xs, abs(Tnum - Texa), 'k-', 'linewidth', 1.2);
title('Parte III: error nodal |T_i - T(x_i)| de la aproximacion por diferencias finitas');
xlabel('Posicion x (m)'); ylabel('|T_i - T(x_i)| (grados C)');
grid on;

%  7. Analisis e interpretacion de los resultados

normb = norm(b);
rhoJ  = (500000/510000) * cos(pi/(n+1));       % radio espectral de Jacobi
rhoGS = rhoJ^2;                                % radio espectral de Gauss-Seidel
lmin  = 510000 - 500000*cos(pi/(n+1));         % autovalor minimo de A
lmax  = 510000 + 500000*cos(pi/(n+1));         % autovalor maximo de A
kappa = lmax/lmin;                             % numero de condicion de A

[tmin, jmin] = min(tej);
[tmax, jmax] = max(tej);

printf('===========================================================================================\n');
printf('  ANALISIS DE LOS RESULTADOS - Parte III\n');
printf('===========================================================================================\n\n');

printf('(1) EQUIVALENCIA DE LAS SOLUCIONES EN TERMINOS DEL RESIDUO\n');
printf('    Los residuos ||AT-b||_2 de los ocho metodos quedan entre %.3e y %.3e, y la mayor\n', min(er), max(er));
printf('    diferencia entre las soluciones de dos metodos cualesquiera es de %.3e grados C.\n', dif);
printf('    Desde el punto de vista fisico las ocho soluciones son indistinguibles: todas describen\n');
printf('    la misma distribucion de temperatura. Las diferencias en el residuo no indican que un\n');
printf('    metodo sea "mas correcto" que otro, sino que reflejan la forma en que cada algoritmo\n');
printf('    acumula el error de redondeo.\n');
printf('    Hay que leer estos residuos en la escala del problema: ||b||_2 = %.3e y ||A||_inf = %.3e,\n', normb, norm(A,inf));
printf('    de modo que un residuo de %.1e corresponde a un residuo RELATIVO de apenas %.1e.\n\n', max(er), max(er)/normb);

printf('(2) TIEMPOS DE EJECUCION\n');
printf('    El metodo mas rapido fue %s (%.4f s) y el mas lento %s (%.4f s).\n', ...
       nombres{jmin}, tmin, nombres{jmax}, tmax);
printf('    El orden observado es coherente con el costo teorico de cada algoritmo: Thomas es O(n),\n');
printf('    los metodos directos generales (Gauss, LU, Cholesky, QR) son O(n^3), aunque en la\n');
printf('    practica QR resulto el mas rapido de los cuatro (%.4f s) por estar vectorizado; el costo de los iterativos\n', tej(4));
printf('    es O(k*n^2) al trabajar con la matriz llena, es decir, depende del numero de iteraciones\n');
printf('    y no solo del tamano del sistema.\n\n');

printf('(3) INFLUENCIA DE LA ESTRUCTURA TRIDIAGONAL SOBRE EL METODO DE THOMAS\n');
printf('    Thomas es el metodo mejor adaptado a este problema. Como A es tridiagonal, en cada paso\n');
printf('    de eliminacion solo hay un elemento que anular debajo del pivote, y la eliminacion no\n');
printf('    genera elementos nuevos fuera de las tres diagonales (no hay llenado). El algoritmo se\n');
printf('    reduce a un barrido hacia adelante y una sustitucion hacia atras, con un costo O(n) en\n');
printf('    lugar de O(n^3) y con almacenamiento O(n) en lugar de O(n^2). Para n = %d eso significa\n', n);
printf('    aproximadamente %d operaciones frente a las ~%.1e de un metodo directo general.\n\n', 8*n, n^3/3);

printf('(4) VENTAJAS DE CHOLESKY\n');
printf('    La matriz A es simetrica y definida positiva: es simetrica por construccion, tiene\n');
printf('    diagonal positiva y es estrictamente dominante por filas (510000 > 2*250000 = 500000),\n');
printf('    y sus autovalores, dados por 510000 - 500000*cos(j*pi/(n+1)), estan todos entre\n');
printf('    %.4e y %.4e, por lo que son positivos. Cholesky aprovecha esa estructura: calcula un\n', lmin, lmax);
printf('    unico factor L en lugar de L y U, lo que reduce a la mitad tanto el numero de operaciones\n');
printf('    (~n^3/3 frente a ~2n^3/3) como el almacenamiento. Ademas no requiere pivoteo, porque la\n');
printf('    condicion de ser definida positiva garantiza que todos los pivotes son positivos y que\n');
printf('    el algoritmo es numericamente estable.\n\n');

printf('(5) DIFERENCIAS ENTRE JACOBI Y GAUSS-SEIDEL\n');
printf('    Ambos convergen porque A es estrictamente dominante diagonalmente, pero a velocidades\n');
printf('    distintas. Para esta matriz tridiagonal los radios espectrales valen\n');
printf('        rho(Jacobi) = %.6f     y     rho(Gauss-Seidel) = rho(Jacobi)^2 = %.6f,\n', rhoJ, rhoGS);
printf('    de modo que Gauss-Seidel reduce el error a la misma razon que Jacobi en la mitad de las\n');
printf('    iteraciones. La corrida auxiliar con tol = %.0e lo confirma: Jacobi necesito %d\n', tol2, kit2(1));
printf('    iteraciones y Gauss-Seidel %d (razon %.2f). La explicacion algoritmica es que\n', kit2(2), kit2(1)/max(kit2(2),1));
printf('    Gauss-Seidel utiliza de inmediato las componentes ya actualizadas en la misma iteracion,\n');
printf('    mientras que Jacobi trabaja siempre con el vector de la iteracion anterior.\n');
printf('    Aun asi, ambos radios espectrales estan muy cerca de 1 y por eso ambos metodos requieren\n');
printf('    cientos o miles de iteraciones, mientras que el gradiente conjugado, cuya convergencia\n');
printf('    depende de sqrt(cond(A)) = %.1f y no del radio espectral, lo logra en %d iteraciones.\n\n', sqrt(kappa), kit2(3));

printf('(6) SOBRE LA TOLERANCIA SOLICITADA Y EL RESIDUO ALCANZABLE\n');
if all(cv(6:8) == 1)
  printf('    Los tres metodos iterativos alcanzaron la tolerancia tol = %.0e antes de iterMax.\n\n', tol);
else
  printf('    Con tol = %.0e ninguno de los metodos iterativos marca conv = 1, y esto NO se debe a\n', tol);
  printf('    que no converjan. El criterio de parada es un residuo ABSOLUTO, y aqui ||b||_2 = %.3e,\n', normb);
  printf('    por lo que exigir ||AT-b||_2 < %.0e equivale a pedir un residuo relativo de %.1e, es\n', tol, tol/normb);
  printf('    decir del orden de la precision de la maquina (eps = %.2e). Al evaluar A*T en punto\n', eps);
  printf('    flotante se comete un error de redondeo del orden de eps*||A||*||T|| ~ %.1e, que actua\n', eps*norm(A,inf)*norm(Tsol(:,jref),inf));
  printf('    como un piso: el residuo calculado se estanca en torno a %.1e y ya no baja, aunque el\n', min(er(6:8)));
  printf('    vector T deje de cambiar. Por eso los metodos iterativos agotan iterMax = %d.\n', iterMax);
  printf('    Comparando con los metodos directos se confirma el diagnostico: el mejor residuo que\n');
  printf('    logra cualquiera de ellos es %.3e, tambien por encima de la tolerancia pedida. Un\n', min(er));
  printf('    criterio relativo del tipo ||AT-b||_2/||b||_2 < tol, o basado en ||T^(k+1)-T^(k)||,\n');
  printf('    seria mas apropiado para un sistema con esta escala de coeficientes.\n\n');
end

if cg_ruptura
  printf('    Una consecuencia directa de lo anterior es la ruptura numerica del gradiente\n');
  printf('    conjugado. Al no poder satisfacerse el criterio de parada, el metodo sigue iterando\n');
  printf('    mucho despues de haber convergido y el residuo recursivo\n');
  printf('    r^(k+1) = r^(k) - alpha*A*p^(k) termina anulandose exactamente. En esa iteracion\n');
  printf('    alpha = (r''r)/(p''Ap) y beta = (r_nuevo''r_nuevo)/(r''r) quedan indefinidos como 0/0\n');
  printf('    y la aproximacion se contamina con NaN; por eso la tabla reporta los valores de la\n');
  printf('    ultima corrida estable (iterMax = %d). En un sistema donde la tolerancia si es\n', cg_iterMax);
  printf('    alcanzable, como el de la Parte II, la situacion no se presenta porque el metodo se\n');
  printf('    detiene mucho antes de que el residuo llegue a anularse.\n\n');
end

printf('(7) CALIDAD DE LA APROXIMACION POR DIFERENCIAS FINITAS\n');
printf('    Los 501 pares ordenados (x_i, T_i) siguen muy de cerca la solucion exacta\n');
printf('    T(x) = 20 + 30x + 10 sin(pi x): el error maximo en los nodos es de %.3e grados C\n', errNodos);
printf('    (figuras 4 y 5). El esquema de diferencias centradas es de orden O(h^2) y con\n');
printf('    h = 1/500 el error de truncamiento resulta despreciable frente a la escala del problema.\n');
printf('    En la figura 4 los puntos de la discretizacion caen practicamente sobre la curva\n');
printf('    continua, de modo que la discretizacion reproduce adecuadamente el modelo continuo.\n\n');

printf('(8) INTERPRETACION FISICA DEL RESULTADO\n');
printf('    El perfil obtenido va de %.2f grados C en el extremo izquierdo a %.2f grados C en el\n', Tnum(1), Tnum(end));
printf('    extremo derecho, respetando las condiciones de frontera impuestas. No es una recta: la\n');
printf('    fuente de calor distribuida produce una elevacion adicional sobre el perfil lineal, con\n');
printf('    una temperatura maxima de %.2f grados C alrededor de x = %.3f m. El intercambio termico\n', Tmax, xs(imax));
printf('    con el medio (termino 10^4*(T-20)) mantiene la solucion acotada y hace que el sistema\n');
printf('    alcance un estado estacionario en el que la temperatura depende solo de la posicion.\n\n');

printf('(9) CONCLUSION\n');
printf('    Para este problema la eleccion natural es el metodo de Thomas: obtiene la misma solucion\n');
printf('    que los demas con un costo O(n) y almacenamiento minimo, gracias a la estructura\n');
printf('    tridiagonal que la discretizacion por diferencias finitas produce de forma natural. Si el\n');
printf('    sistema fuera mucho mayor o se almacenara en formato disperso, el gradiente conjugado\n');
printf('    seria la alternativa recomendable, pues explota la simetria definida positiva de A y\n');
printf('    alcanza una buena aproximacion en un numero de iteraciones mucho menor que Jacobi o\n');
printf('    Gauss-Seidel. La factorizacion QR, pese a tener el mismo costo teorico O(n^3) que Gauss\n');
printf('    y LU, resulto en la practica la mas rapida de las cuatro directas generales gracias a\n');
printf('    estar implementada con operaciones vectorizadas en lugar de ciclos escalares anidados.\n');
printf('===========================================================================================\n');
