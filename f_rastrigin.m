function F = f_rastrigin(X)
%F_RASTRIGIN Summary of this function goes here
%   Detailed explanation goes here

n = size(X, 2);

A = 10;
F = A*n + sum((X .^ 2) - (A * cos(2 * pi * X)), 2);

F = F';

end
