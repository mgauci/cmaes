clear

n = 100;


opts.f_evals_max = (10 ^ 4) * n;
opts.lambda = 1000;


opt = cmaes(-10 + (20 * rand(1, n)), 0.3 * 20, opts);

stop = 0;
g = 0;

tic
while (~stop)
    g = g + 1;
    solutions = opt.ask();
    fitnesses = f_rastrigin(solutions);
    stop = opt.tell(fitnesses);
    F = f_rastrigin(opt.get_m());
    disp(F);
end
toc

