clear

n = 2;
G = 1000;

% opts.lambda = 5;
opts.f_evals_max = 2 * 10 ^ 4;

opt = cmaes(-10 + (20 * rand(1, n)), 0.3 * 20, opts);

F = [];
stop = 0;
g = 0;

tic
while (~stop)
    g = g + 1;
    solutions = opt.ask();
    fitnesses = f_rastrigin(solutions);
    stop = opt.tell(fitnesses);

    F = [F, f_rastrigin(opt.get_m())];
end
toc

F(length(F))

plot(F)

