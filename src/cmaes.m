classdef cmaes < handle
    %CMAES Covariance Matrix Adaptation Evolution Strategy Optimizer
    %   Detailed explanation goes here
    
    properties
        n;          % Problem dimension.
        lambda;     % Population size before selection.
        mu;         % Population size after selection.
        w;          % Recombination weights.
        c_sigma;    % Cumulation learning rate for step size update.
        d_sigma;    % Damping parameter for step size update.
        c_c;        % Cumulation learning rate for rank-one update of
                    % covariance matrix.
        c_1;        % Learning rate for rank-one update of covariance
                    % matrix.
        c_mu;       % Learning rate for rank-mu update of covariance 
                    % matrix.
        
        m;         % Weighted mean.
        sigma;     % Overall step size.
        
        p_sigma;
        p_c;
        
        C;         % Covariance matrix.
        B;         % Eigenvector matrix of C.
        D;         % Diagonal matrix of square roots of eigenvalues of C.
        
        Y;         % Offspring population (before mean-shifting).
        Y_sel;
        y_w;       % Weighted average of all y's.
        
        g;         % Generation.
        
        mu_eff;    % Variance effective selection mass. 
        EN0I;      % Expectation of randomly-generated vector according to
                   % Normal(0, I) distribution.
        
        const_40_1;
        const_40_2;
        const_41_1;
        const_42_1;
        const_42_2;
        const_43_1;
        
        h_sigma;
        delta_h_sigma;
        
        const_delta_h_sigma;
        const_h_sigma_threshold;
        
        f_evals = 0; % Number of function evaluations.
        
        f_evals_max;
        
    end
    
    methods
        function obj = cmaes(m, sigma, opts)
            assert(isrow(m));
            assert(isscalar(sigma));
            assert(sigma > 0);
            
            obj.m = m;
            obj.sigma = sigma;
            
            obj.init_params(opts);
            obj.init_consts();
            obj.init_vars();
            
            if isfield(opts, 'f_evals_max')
                obj.f_evals_max = opts.f_evals_max;
            end
        end
        
        function [] = init_vars(obj)
            obj.g = 0;
            obj.p_sigma = zeros(1, obj.n);
            obj.p_c = zeros(1, obj.n);
            obj.C = eye(obj.n);
        end

        
        function [] = init_params(obj, opts)
            obj.n = length(obj.m);
            
            % Eq. 44.
            if isprop(opts, 'lambda')
                obj.lambda = opts.lambda;
            else
                obj.lambda = 4 + floor(3 * log(obj.n));
            end
            mu_prime = obj.lambda / 2;
            obj.mu = floor(mu_prime);
            
            % Eq. 45.
            i = 1 : obj.mu;
            obj.w = log(mu_prime + 0.5) - log(i);
            obj.w = obj.w / sum(obj.w);
            
            obj.mu_eff = 1 / sum(obj.w .^ 2);
            
            % Eq. 46.
            obj.c_sigma = (obj.mu_eff + 2) / (obj.n + obj.mu_eff + 5);
            obj.d_sigma = 1 + (2 * max(0, sqrt((obj.mu_eff - 1) / ...
                (obj.n + 1)) - 1)) + obj.c_sigma;
            
            % Eq. 47.
            obj.c_c = (4 + (obj.mu_eff / obj.n)) / ...
                (obj.n + 4 + (2 * (obj.mu_eff / obj.n)));
            
            % Eq. 48.
            obj.c_1 = 2 / (((obj.n + 1.3) .^ 2)  + obj.mu_eff);
            
            % Eq. 49
            alpha_mu = 2;
            obj.c_mu = min(1 - obj.c_1, ...
                alpha_mu * ((obj.mu_eff - 2 + (1 / obj.mu_eff)) / ...
                (((obj.n + 2) .^ 2) + (alpha_mu * (obj.mu_eff / 2)))));
        end
        
        function [] = init_consts(obj)
            % --- Pre-computed constants that appear in equations. ---
            
            obj.const_40_1 = 1 - obj.c_sigma;
            obj.const_40_2 = sqrt(obj.c_sigma * (2 - obj.c_sigma) * ...
                obj.mu_eff);
            
            obj.const_41_1 = obj.c_sigma / obj.d_sigma;
            
            obj.const_42_1 = 1 - obj.c_c;
            obj.const_42_2 = sqrt(obj.c_c * (2 - obj.c_c) * obj.mu_eff);
            
            obj.const_43_1 = 1 - obj.c_1 - obj.c_mu;
            
            obj.EN0I = sqrt(obj.n) * (1 - (1 / (4 * obj.n)) + ...
                (1 / (21 * (obj.n ^2 ))));
            
            obj.const_h_sigma_threshold = (1.4 + (2 / (obj.n + 1))) * ...
                obj.EN0I;
            
            obj.const_delta_h_sigma = obj.c_c * (2 - obj.c_c);
        end
        
        function solutions = ask(obj)
            [obj.B, D2] = eig(obj.C);
            obj.D = diag(sqrt(diag(D2))); % Faster than obj.D = sqrt(D2);
            
            Z = normrnd(0, 1, obj.n, obj.lambda);
            
            obj.Y = (obj.B * obj.D * Z).';
            
            solutions = repmat(obj.m, obj.lambda, 1) + (obj.sigma * obj.Y);
        end
        
        function stop = tell(obj, fitnesses)
            assert(isrow(fitnesses)); 
            assert(length(fitnesses) == obj.lambda);
            
            obj.g = obj.g + 1;
            obj.f_evals = obj.f_evals + obj.lambda;
            
            [~, indeces] = sort(fitnesses);
            
            % Selection and recombination.
            % Eq. 38.
            obj.Y_sel = obj.Y(indeces(1 : obj.mu), :);
            obj.y_w = sum(bsxfun(@times, obj.w.', obj.Y_sel));
            
            % Eq. 39.
            obj.m = obj.m + (obj.sigma * obj.y_w);
            
            % Step-size control.
            % Eq. 40.
            C_minus_half = obj.B * diag(1 ./ diag(obj.D)) * obj.B';
            obj.p_sigma = (obj.const_40_1 * obj.p_sigma) + ...
                (obj.const_40_2 * obj.y_w * C_minus_half);
            
            % Eq. 41.
            obj.sigma = obj.sigma * exp(obj.const_41_1 * ...
                ((norm(obj.p_sigma) / obj.EN0I) - 1));
            
            % Eq. 42.
            obj.update_h_sigma();
            obj.p_c = (obj.const_42_1 * obj.p_c) + ...
                (obj.h_sigma * obj.const_42_2 * obj.y_w);
            
            % Eq. 43.
            obj.update_delta_h_sigma();
            S = obj.Y_sel.' * bsxfun(@times, obj.Y_sel, obj.w(:));         
            obj.C = (obj.const_43_1 * obj.C) + (obj.c_1 * ...
                ((obj.p_c.' * obj.p_c) + ...
                (obj.delta_h_sigma * obj.C))) + (obj.c_mu * S);
            obj.C = triu(obj.C) + triu(obj.C, 1).';
            
            stop = obj.check_stop_conds();
        end
        
        function stop = check_stop_conds(obj)
            stop = 0;
            if (obj.f_evals > obj.f_evals_max)
                stop = 1;
            end
        end
        
        function [] = update_h_sigma(obj)
            if ((norm(obj.p_sigma) / sqrt(1 - (obj.const_40_1 ...
                    ^ (2 * (obj.g + 1))))) < obj.const_h_sigma_threshold)
                obj.h_sigma = 1;
            else
                obj.h_sigma = 0;
            end
        end
        
        function [] = update_delta_h_sigma(obj)
            obj.delta_h_sigma = (1 - obj.h_sigma) * ...
                obj.const_delta_h_sigma;
        end
        
        function m = get_m(obj)
            m = obj.m;
        end
    end
    
end

