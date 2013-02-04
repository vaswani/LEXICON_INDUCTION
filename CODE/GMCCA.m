classdef GMCCA
    
    methods(Static)
        
        function F=find_matching(options, data)
            % computes the within graph aided mCCA between X and Y.
            % returns pi, the permutation that matches samples (rows) of X
            % with rows of Y.
            if options.use_single == 1
                data.X.features = single(data.X.features); % work w/ single precision.
                data.Y.features = single(data.Y.features);
            end
            [NX,DX] = size(data.X.features);
            [NY,DY] = size(data.Y.features);
            assert(NX==size(data.X.G,1));
            assert(NY==size(data.Y.G,1));
            
            % TODO: 
            % Should initialize with something reasonable. not the
            % identity.
            % Options should have lambda, and length or random walk K.
            F.hamming = inf(options.T, 1);
            pi_t = 1:NY;
            for t=1:options.T,
                options.t = t;
                inv_pi = Util.inverse_perm(pi_t);
                Z      = CCAUtil.latentCCA(data.X.features, data.Y.features(pi_t,:), options);  % compute latent representation under matching
                Z.Y    = Z.Y(inv_pi,:);                                                         % permute back.
                W_t    = MatchingUtil.makeWeights(options, Z.X, Z.Y, data.X.G, data.Y.G);       % compute weights 
                old_pi = pi_t;
                tic; [pi_t, cost] = MatchingUtil.match(W_t, options); toc;                      % compute matching
                % pi_t   = Util.randswap(pi_t, 4);
                Util.is_perm(pi_t); %% assert pi_t is a valid permutation
                
                F.obj(t) = cost;
                F.hamming(t) = Util.hamming(pi_t, old_pi);
                if isfield(data, 'true_pi')   % compute distance to true permutation.
                    F.hamming(t) = Util.hamming(Util.inverse_perm(pi_t), data.true_pi); 
                    fprintf('%d), norm=%2.3f\thamming=%f\n',t, F.obj(t), F.hamming(t));
                else
                    fprintf('%d), norm=%2.3f, hamming change=%d\n',t, F.obj(t), F.hamming(t));
                end
                
                if F.hamming(t)==0  % fixed point - stopping condition
                    fprintf('Fixed point after t=%d iterations\n', t);
                    break;
                    %options.d = options.d + 10; % this will probably throw2 a bug later.
                end
            end
%             if any(F.hamming ~= inf)
%                 plot(F.hamming);
%             end
            F.pi = pi_t;
        end
        
        function options = makeOptions(weight_type, T, d, K, lambda)
            options.T = T;
            options.weight_type = weight_type;
            options.d = d;
            options.K = K;
            options.lambda = lambda;
            options.use_single = 0;
        end
        
        function run(maxN)
            %% DATA
            %source.filename = './data/en.ortho.v1_en.syns.v1.mat';
            %target.filename = './data/es.ortho.v1_es.syns.v1.mat';
            source.filename = './data/FEB3_en.features.10k_en.syns.v1.mat';
            target.filename = './data/FEB3_es.features.10k_es.syns.v1.mat';
            
            source = Common.loadMat(source);
            target = Common.loadMat(target);
            [~, source.pi] = Common.getFreq(source);
            [~, target.pi] = Common.getFreq(target);
            
            data.X = GMCCA.setup_most_frequent(source, maxN);
            data.Y = GMCCA.setup_most_frequent(target, maxN);
            
            % figure out initial matching.
            % for now, based on edit-distance
            match = MatchingUtil.init_matching(data.X.words, data.Y.words, 3);
            data.seed.match = match;
            data.seed.N = length(match.source);
            
            [data.X] = GMCCA.fix_matched_words(data.X, match.source);
            [data.Y] = GMCCA.fix_matched_words(data.Y, match.target);
            %init_alignment = GMCCA.getAlignment(data.X.words, data.Y.words)
            
            data.source = source;
            data.target = target;
            
            %% OPTIONS
            weight_type = 'inner';
            T = 20;  % at most 20 iterations
            p = 0.7; % start with d that preserves at least p of the eigenmass of X and Y
            d = max(Util.mass_to_dim(data.X.features, p), Util.mass_to_dim(data.Y.features, p)); % use d correlation dims
            K = 1;   % random walk steps
            lambda = 0; % diffusion rate
            options = GMCCA.makeOptions(weight_type, T, d, K,lambda); 
            
            
            fprintf('rank(X)=%d, rank(Y)=%d\n',rank(data.X.features), rank(data.Y.features));
            F = GMCCA.find_matching(options, data);
            
            alignment = GMCCA.getAlignment(data.X.words, data.Y.words, F.pi)
        end
        
        function X = setup_most_frequent(source, maxN)
            source.pi = source.pi(1:maxN);
            pi = source.pi;
            X.features      = source.features(pi,:);
            X.features(:,1) = log10(X.features(:,1));
            X.words = source.words(pi);
            X.G     = Util.to_stochastic_graph(source.G(pi, pi));
        end
        
        function alignment = getAlignment(wordsX, wordsY, pi)
            N = length(wordsX);
            if nargin < 3
            	pi = 1:N;
            end
            alignment = [ (mat2cell([1:N]', ones(N,1))),wordsX, wordsY(pi)];
        end
        
        function X = fix_matched_words(X, pi)
            Nw = length(X.words);
            
            rest = setdiff(1:Nw, pi);
            new_order = [pi, rest];
            
            % move matched words to the beginning.
            X.words = X.words(new_order);
            X.features = X.features(new_order,:);
            X.G = X.G(new_order,new_order);
        end
        
        function recoverd = sanityCheck(seed, noise_coeff, lambda_coeff)
            if nargin == 0
                seed = 3;
                noise_coeff = 0.1;
                lambda_coeff = 1;
            end
            rng(seed);
            % create data
            N = 500; 
            D = 40;
            data_type = 1; % 0=mock data with empty graph, 1=mock data with sparse graph
            data = GMCCA.loadMockData(data_type, N, D, noise_coeff); 
            % create options
            T = 10; % at most 200 iterations
            d = 30;  % use 30 correlation dims
            weight_type = 'inner'; % inner product similarity
            K = 1;
            lambda = noise_coeff*10*lambda_coeff;
            options = GMCCA.makeOptions(weight_type, T, d, K, lambda); 
            F=GMCCA.find_matching(options, data);
            GMCCA.outputAlignment(data.X.words, data.Y.words, F.pi);
            
            recoverd = all([data.X.words{:}] == [data.Y.words{F.pi}]);
        end
        
        function plot_prob_of_recovery()
            noise_coeffs = 0:0.1:0.7;
            N_NC = length(noise_coeffs);
            rec.base  = zeros(N_NC,1);
            rec.graph = zeros(N_NC,1);
            T = 10;
            
            for i = 1:N_NC
                nc = noise_coeffs(i);
                for seed = 1:T,
                    rec.base(i)  = rec.base(i) + GMCCA.sanityCheck(seed,nc,0);
                    rec.graph(i) = rec.graph(i)+ GMCCA.sanityCheck(seed,nc,1);
                end
            end
            plot([rec.base, rec.graph]/T);
        end
    end
    
    methods(Static, Access=private);
        
        function v=hamming(p,q)
            v=sum(p~=q);
        end
        
        % random swap k elements in pi
        function pi=randswap(pi, k)
            q = pi;
            i = randperm(length(q));
            i = i(1:k); % take k distinct elements.
            if k==2
                pi(i) = pi(i(end:-1:1));
            else
                pi(i) = q(sort(i));
            end
        end
        
        function F = random_rotate_shift(F)
            D = size(F,2);
            [rand_rot,~] = svd(rand(D)); % random rotation
            rand_mean = rand(1,D);
            F = F * rand_rot;
            F = bsxfun(@plus, F, rand_mean); % shift randomly.
        end
        
        function data = loadMockData(type, N, D, noise_coeff)
            data.X.words = mat2cell(1:N,1,ones(N,1));
            Z = randn(N, D);         % generate random gaussian data
            if type == 0                             %% Create mock data    
                data.X.G = zeros(N);                 % create empty graphs.
            elseif type == 1 
                % mock data with graphs
                data.X.G = rand(N,N)<0.01;
            else
                error('unknown type %d\n', type);
            end
            
            data.X.features = Z + noise_coeff*randn(N,D);                % Y = X + noise
            data.Y.features = Z + noise_coeff*randn(N,D);                % Y = X + noise
            q = 0.92;
            true_pi         = Util.randswap(1:N, q*N);     % permute Y
            data.Y.features = data.Y.features(true_pi, :);
            data.Y.words    = data.X.words(true_pi);
            
            data.Y.G = data.X.G(true_pi, true_pi);
            data.X.G = and(data.X.G, (rand(N,N)>0.1)); % remove a few randomly.
            data.Y.G = and(data.Y.G, (rand(N,N)>0.1));
            data.X.G = Util.to_stochastic_graph(data.X.G);
            data.Y.G = Util.to_stochastic_graph(data.Y.G);
            
            data.X.features = GMCCA.random_rotate_shift(data.X.features);
            data.Y.features = GMCCA.random_rotate_shift(data.Y.features);

            data.true_pi = true_pi;                 % figure out inverse permutation
        end
    end
end

