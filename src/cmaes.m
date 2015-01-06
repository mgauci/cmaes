classdef cmaes < handle
    %CMAES Covariance Matrix Adaptation Evolution Strategy
    %   TODO: Detailed documentation.
    
    properties
        %-----------------------------------------------------------------------
        % Algorithm paramters.
        %-----------------------------------------------------------------------
        n;          % Problem dimension.
        lambda;     % Population size before selection.
        mu;         % Population size after selection.
        w;          % Recombination weights.
        c_sigma;    % Cumulation learning rate for step size update.
        d_sigma;    % Damping parameter for step size update.
        c_c;        % Cumulation learning rate for for rank one update
                    % of covariance matrix.
        c_1;        % Learning rate for rank-one update of covariance
                    % matrix.
        c_mu;       % Learning rate for rank-mu update of covariance 
                    % matrix.
                    
        %-----------------------------------------------------------------------
        % Termination parameters.
        %-----------------------------------------------------------------------
        f_evals_max;    % Maximum number of objective function evaluations.      
        
        %-----------------------------------------------------------------------
        % Algorithm variables.                                                      
        %-----------------------------------------------------------------------
        g;          % Generation.
        Y;          % Candidate solutions (realizations from a multivariate
                    % normal distribution with zero mean and covariance
                    % matrix C).
        Y_sel;      % Selected candidate solutions ordered by fitness. 
        y_w;        % Step of distribution mean disregarding step size.
        m;          % Weighted mean of selected points.
        p_sigma;    % Conjugate evolution path.
        sigma;      % Overall step size (standard deviation).
        p_c;        % Evolution path.
        C;          % Covariance matrix.
        B;          % Eigenvector matrix of C.
        D;          % Diagonal matrix of square roots of eigenvalues of C.
        
        h_sigma;          % Todo: potentially change to local variables.
        delta_h_sigma;    % """
        
        %-----------------------------------------------------------------------
        % Termination variables.                                                    
        %-----------------------------------------------------------------------
        f_evals = 0;    % Number of function evaluations.
        
        %-----------------------------------------------------------------------
        % Pre-computed constants.                                                      
        %-----------------------------------------------------------------------
        mu_eff;        % Variance effective selection mass. 
        EN0I;          % Expectation of Euclidean norm of N(0, I) 
                       % distributed random vector.
                       
        const_40_1;    %
        const_40_2;    % Pre-computed constants that appear in equations.
        const_41_1;    % First numbers correspond to equation numbers in
        const_42_1;    % [1]. Second numbers enumerate different constants
        const_42_2;    % within a single equation.
        const_43_1;    %

        delta_h_sigma_value;
        h_sigma_threshold;
    end
    
    methods
        function obj = cmaes(m, sigma, opts)
            obj.validate_m_and_sigma(m, sigma); 
            
            obj.n = length(m);
            obj.m = m;
            obj.sigma = sigma;
            
            obj.validate_opts(opts);
            
            obj.init_params(opts);
            obj.init_consts();
            obj.init_vars();
            
            if isfield(opts, 'f_evals_max')
                obj.f_evals_max = opts.f_evals_max;
            end
        end
        
        function validate_m_and_sigma(~, m, sigma)
            if ~(isvector(m) && isnumeric(m) && isreal(m))
                error('m must be a vector of real numbers.');
            end
            if ~(isscalar(sigma) && isnumeric(sigma) && isreal(sigma) ...
                    && (sigma > 0))
                error('sigma must be a positive real scalar.');
            end
        end
        
        function validate_opts(obj, opts)
            if isfield(opts, 'lambda')
                if ((~isscalar(opts.lambda)) || ...
                        (~isnumeric(opts.lambda)) || ...
                        (~isreal(opts.lambda)))
                    error('lambda must be an integer.');
                end
                if (~(opts.lambda >= 2))
                    error('lambda must be an greater than or equal to 2.');
                end
                if (~(opts.lambda == floor(opts.lambda)))
                    warning('lambda is not an integer. Taking floor value.');
                end
                local_lambda = opts.lambda;
            else
                local_lambda = 4 + floor(3 * log(obj.n));
            end
            
            if isfield(opts, 'mu')
                if ((~isscalar(opts.mu)) || (~isnumeric(opts.mu)) || ...
                        (~isreal(opts.mu)))
                    error('mu must be an integer.');
                end
                if (~(opts.mu >= 1))
                    error('mu must be greater than or equal to 1.');
                end
                if (~(opts.mu < local_lambda))
                    error('mu must be smaller than lambda.');
                end
                if (~(opts.mu == floor(opts.mu)))
                    warning ('mu is not an integer. Taking floor value.');
                end
                local_mu = opts.mu;
            else
                local_mu = floor(local_lambda / 2);
            end
            
            if (isfield(opts, 'w'))
                if ((~(isvector(opts.w))) || (~(isnumeric(opts.w))) || ...
                        (~isreal(opts.mu)))
                    error('w must be a vector of real numbers.');
                end
                if (~(length(opts.w) == local_mu))
                    error('w must have length mu.');
                end
                if (~(min(opts.w) > 0))
                    error('w must contain only numbers greater than or equal to 0.');
                end
                if (~(max(opts.w) > 0))
                    error('w must contain at least one number greater than 0.');
                end
                if (~(sum(opts.w) == 1))
                    warning('sum(w) is not equal to 1. Normalizing.');
                end
            end
            
            if isfield(opts, 'c_sigma')
                if ((~isscalar(opts.c_sigma)) || ...
                        (~isnumeric(opts.c_sigma)) || ...
                        (~isreal(opts.c_sigma)))
                    error('c_sigma must be a scalar real number.');
                end
                if ((~(0 <= opts.c_sigma)) || (~(opts.c_sigma <= 1)))
                    error('c_sigma must be in [0, 1].');
                end
            end
            
            if isfield(opts, 'd_sigma')
            end
            
            if isfield(opts, 'c_c')
            end
            
            if isfield(opts, 'c_1')
            end
            
            if isfield(opts, 'c_mu')
            end
            
            if isfield(opts, 'f_evals_max')
                if (~isscalar(opts.f_evals_max'))
                    error('f_evals_max must be a scalar.');
                end
            end
        end % Function validate_opts().
        
        function [] = init_vars(obj)
            obj.g = 0;
            obj.p_sigma = zeros(1, obj.n);
            obj.p_c = zeros(1, obj.n);
            obj.C = eye(obj.n);
        end

        
        function [] = init_params(obj, opts)
            % Eq. 44.
            if isfield(opts, 'lambda')
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
            obj.const_40_1 = 1 - obj.c_sigma;
            obj.const_40_2 = sqrt(obj.c_sigma * (2 - obj.c_sigma) * ...
                obj.mu_eff);
            obj.const_41_1 = obj.c_sigma / obj.d_sigma;
            obj.const_42_1 = 1 - obj.c_c;
            obj.const_42_2 = sqrt(obj.c_c * (2 - obj.c_c) * obj.mu_eff);
            obj.const_43_1 = 1 - obj.c_1 - obj.c_mu;
            
            obj.EN0I = sqrt(obj.n) * (1 - (1 / (4 * obj.n)) + (1 / ...
                (21 * (obj.n ^2 ))));
            obj.h_sigma_threshold = (1.4 + (2 / (obj.n + 1))) * obj.EN0I;
            obj.delta_h_sigma_value = obj.c_c * (2 - obj.c_c);
        end
        
        function solutions = ask(obj)
            [obj.B, D2] = eig(obj.C);
            obj.D = diag(sqrt(diag(D2))); % Faster than obj.D = sqrt(D2);
            
            Z = normrnd(0, 1, obj.n, obj.lambda);
            
            obj.Y = (obj.B * obj.D * Z).';
            
            solutions = repmat(obj.m, obj.lambda, 1) + (obj.sigma * obj.Y);
        end
        
        function [] = selection_and_recombination(obj)
            % Eq. 38.
            obj.Y_sel = obj.Y(indeces(1 : obj.mu), :);
            obj.y_w = sum(bsxfun(@times, obj.w.', obj.Y_sel));
            
            % Eq. 39.
            obj.m = obj.m + (obj.sigma * obj.y_w);
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
                    ^ (2 * (obj.g + 1))))) < obj.h_sigma_threshold)
                obj.h_sigma = 1;
            else
                obj.h_sigma = 0;
            end
        end
        
        function [] = update_delta_h_sigma(obj)
            obj.delta_h_sigma = (1 - obj.h_sigma) * ...
                obj.delta_h_sigma_value;
        end
        
        function m = get_m(obj)
            m = obj.m;
        end
    end
    
end

