clc;
clear;
close all;

%funciones de parte1.m
M = parte1();

%Matriz A y el vector b
% A es de 300x300
N = 300;
b = ones(N, 1);
A = 4 * eye(N) - diag(ones(N-1, 1), 1) - diag(ones(N-1, 1), -1);

x0 = zeros(N, 1);
tol = 1e-8;
iterMax = 10000;


%Métodos Directos

%Eliminación Gaussiana
tic;
x_gauss = M.eliminacion_gaussiana(A, b);
t_gauss = toc;
e_gauss = norm(A*x_gauss - b);

%Factorización LU
tic;
x_lu = M.factorizacion_LU(A, b);
t_lu = toc;
e_lu = norm(A*x_lu - b);

%Cholesky
tic;
x_chol = M.cholesky(A, b);
t_chol = toc;
e_chol = norm(A*x_chol - b);

%Factorización QR
tic;
x_qr = M.factorizacion_QR(A, b);
t_qr = toc;
e_qr = norm(A*x_qr - b);

%Método de Thomas
tic;
x_thom = M.metodo_thomas(A, b);
t_thom = toc;
e_thom = norm(A*x_thom - b);

% 5. Ejecución de Métodos Iterativos

%Jacobi
tic;
[x_jac, e_jac, k_jac, conv_jac] = M.jacobi(A, b, x0, tol, iterMax);
t_jac = toc;

%Gauss-Seidel
tic;
[x_gs, e_gs, k_gs, conv_gs] = M.gauss_seidel(A, b, x0, tol, iterMax);
t_gs = toc;

%Gradiente Conjugado
tic;
[x_gc, e_gc, k_gc, conv_gc] = M.gradiente_conjugado(A, b, x0, tol, iterMax);
t_gc = toc;

% Tabla Comparativa
fprintf('\n========================================================================\n');
fprintf('%-22s | %-12s | %-12s | %-7s | %-5s\n', 'Método', 'Error', 'Tiempo (s)', 'Iters', 'Conv');
fprintf('------------------------------------------------------------------------\n');
fprintf('%-22s | %-12.4e | %-12.6f | %-7s | %-5s\n', 'Eliminación Gaussiana', e_gauss, t_gauss, '-', '-');
fprintf('%-22s | %-12.4e | %-12.6f | %-7s | %-5s\n', 'Factorización LU', e_lu, t_lu, '-', '-');
fprintf('%-22s | %-12.4e | %-12.6f | %-7s | %-5s\n', 'Cholesky', e_chol, t_chol, '-', '-');
fprintf('%-22s | %-12.4e | %-12.6f | %-7s | %-5s\n', 'Factorización QR', e_qr, t_qr, '-', '-');
fprintf('%-22s | %-12.4e | %-12.6f | %-7s | %-5s\n', 'Método de Thomas', e_thom, t_thom, '-', '-');
fprintf('%-22s | %-12.4e | %-12.6f | %-7d | %-5d\n', 'Jacobi', e_jac, t_jac, k_jac, conv_jac);
fprintf('%-22s | %-12.4e | %-12.6f | %-7d | %-5d\n', 'Gauss-Seidel', e_gs, t_gs, k_gs, conv_gs);
fprintf('%-22s | %-12.4e | %-12.6f | %-7d | %-5d\n', 'Gradiente Conjugado', e_gc, t_gc, k_gc, conv_gc);
fprintf('========================================================================\n\n');

%Gráficas

metodos_nombres = {'Gauss', 'LU', 'Cholesky', 'QR', 'Thomas', 'Jacobi', 'G-Seidel', 'Grad. Conj.'};
iterativos_nombres = {'Jacobi', 'Gauss-Seidel', 'Grad. Conj.'};

% Gráfica de Errores
figure('Name', 'Comparación de Errores');
errores = [e_gauss, e_lu, e_chol, e_qr, e_thom, e_jac, e_gs, e_gc];
bar(errores, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', metodos_nombres, 'YScale', 'log');
ylabel('Error ||Ax-b||_2');
title('Error por Método Numérico (Escala Logarítmica)');
grid on;

% Gráfica de tiempos de ejecución
figure('Name', 'Comparación de Tiempos');
tiempos = [t_gauss, t_lu, t_chol, t_qr, t_thom, t_jac, t_gs, t_gc];
bar(tiempos, 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'XTickLabel', metodos_nombres, 'YScale', 'log');
ylabel('Tiempo de ejecución (s)');
title('Tiempo de Ejecución por Método (Escala Logarítmica)');
grid on;

% Gráfica de Número de Iteraciones
figure('Name', 'Comparación de Iteraciones');
iteraciones = [k_jac, k_gs, k_gc];
bar(iteraciones, 'FaceColor', [0.3 0.7 0.4]);
set(gca, 'XTickLabel', iterativos_nombres);
ylabel('Número de iteraciones');
title('Iteraciones requeridas (Métodos Iterativos)');
grid on;

%Análisis Comparativo
fprintf('================ ANÁLISIS COMPARATIVO DE RESULTADOS ====================\n');
fprintf('1. Métodos Directos vs. Estructura de la Matriz:\n');
fprintf('   El Método de Thomas obtiene el menor tiempo de ejecución entre los \n');
fprintf('   métodos directos. Esto se debe a que la matriz A es tridiagonal, \n');
fprintf('   y Thomas explota esta estructura reduciendo el costo computacional \n');
fprintf('   a O(n) operaciones. Por el contrario, la Factorización QR es el \n');
fprintf('   método más lento, ya que el algoritmo de Gram-Schmidt requiere \n');
fprintf('   O(n^3) operaciones, ignorando completamente los ceros de la matriz.\n\n');

fprintf('2. Condiciones de la Matriz A:\n');
fprintf('   La matriz A es Simétrica Definida Positiva, ya que es \n');
fprintf('   estrictamente diagonalmente dominante por filas (|4| > |-1| + |-1|) \n');
fprintf('   y simétrica. Esto garantiza teóricamente el éxito de la \n');
fprintf('   factorización de Cholesky y la convergencia de los métodos de \n');
fprintf('   Jacobi, Gauss-Seidel y Gradiente Conjugado.\n\n');

fprintf('3. Métodos Iterativos:\n');
fprintf('   - Gauss-Seidel converge más rápido y en menos iteraciones que Jacobi \n');
fprintf('     porque utiliza los valores actualizados de x^(k+1) inmediatamente \n');
fprintf('     durante el cálculo.\n');
fprintf('   - El Gradiente Conjugado es mejor computacionalmente a Jacobi y \n');
fprintf('     Gauss-Seidel. Al ser un método de subespacios de Krylov \n');
fprintf('     aplicado a una matriz SPD, encuentra la dirección óptima \n');
fprintf('     minimizando el error en muchas menos iteraciones, lo que lo \n');
fprintf('     hace el iterativo más robusto y rápido para este problema.\n');
fprintf('========================================================================\n');
