function fitnesses = sphere_function(X)
    fitnesses = sum(X .^ 2, 2)';
end

