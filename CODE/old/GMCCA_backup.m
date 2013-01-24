classdef GMCCA_backup
    
    methods(Static)
        
        function agg_pi=find_matching(options, data)
            % computes the within graph aided mCCA between X and Y.
            % returns pi, the permutation that matches samples (rows) of X
            % with rows of Y.
            if options.use_single == 1
                data.X = single(data.X); % work w/ single precision.
                data.Y = single(data.Y);
            end
            [NX,DX] = size(data.X);
            [NY,DY] = size(data.Y);
            assert(NX==size(data.GX,1));
            assert(NY==size(data.GY,1));
            
            % TODO: 
            % Should initialize with something reasonable. not the
            % identity.
            % Options should have lambda, and length or random walk K.
            hamming = inf(options.T, 1);
            agg_pi = 1:NY;
            options.d = min(30, options.max_latent_dim);
            for t=1:options.T,
                Z         = CCAUtil.latentCCA(data.X, data.Y, options);                     % compute latent representation under matching
                W_t       = MatchingUtil.makeWeights(options, Z.X, Z.Y, data.GX, data.GY);  % compute weights 
                pi_t      = MatchingUtil.match(options, W_t);                               % compute matching
                assert(length(pi_t)==length(unique(pi_t)));                                 % assert pi is indeed a permutation.
                pi_t      = Util.randswap(pi_t, 4);
                
                data.Y  = data.Y(pi_t, :);        % permute according to pi_t
                data.GY = data.GY(pi_t, pi_t);    % permute the graph too
                agg_pi = agg_pi(pi_t);              % compute the aggregate permutation
                
                if isfield(data, 'inv_true_pi')   % compute distance to true permutation.
                    hamming(t) = Util.hamming(agg_pi, data.inv_true_pi); 
                    fprintf('%d), norm=%f\thamming=%f\n',t, norm(data.X-data.Y), hamming(t));
                else
                    % this is not really the objective function,
                    % but it should generally increase (??)
                    obj = Z.X .* Z.Y(pi_t,:);
                    obj = sum(obj(:));
                    fprintf('%d), norm=%f\n',t, obj/options.d);
                end
                
                % TODO: take note of 
                % what to do when NX != NY ? does the matching still
                % work?
                if all(pi_t == 1:length(pi_t))  % fixed point - stopping condition
                    fprintf('Fixed point after t=%d iterations\n', t);
                    options.d = options.d + 1;
                    if options.d > options.max_latent_dim
                        break;
                    end
                end
            end
            if any(hamming ~= inf)
                plot(hamming)
            end
        end
        
        function [obj] = objective(options, Z, GX,GY, pi)
            obj.inner = Z.X.*Z.Y(pi,:);
            %% 
            % TODO: change the algorithm such that the only thing
            % that we only keep a global permutation and the data.X, data.Y
            % remain fixed.
            obj.graph = 0;
            if isempty(options.weight_type) || strcmpi(options.weight_type, 'inner')
                for k=1:options.K,
                    W = W + options.lambda^k*(GX^k)*U*(GY^k)';
                    %obj.graph = obj.graph + options.lambda^k*(GX^k)*
                end
            end
            
            obj.val = obj.inner + obj.graph;
        end
        
        function options=makeOptions(weight_type, T, d, K, lambda)
            options.T = T;
            options.weight_type = weight_type;
            options.max_latent_dim = d;
            options.K = K;
            options.lambda = lambda;
            options.use_single = 1;
        end
        
        function run(maxN)
            source.filename = './data/en.ortho.v1_en.syns.v1.mat';
            target.filename = './data/es.ortho.v1_es.syns.v1.mat';
            
            source = Common.loadMat(source);
            target = Common.loadMat(target);
            
            weight_type = 'inner';
            T = 20; % at most 200 iterations
            d = 50;  % use d correlation dims
            K = 1;   % random walk steps
            lambda = 2; % diffusion rate
            options = GMCCA_backup.makeOptions(weight_type, T, d, K,lambda); 
            
            [~, pi.X] = Common.getFreq(source);
            [~, pi.Y] = Common.getFreq(target);
            pi.X = pi.X(1:maxN);
            pi.Y = pi.Y(1:maxN);
            
            data.X      = source.features(pi.X,:);
            data.X(:,1) = log10(data.X(:,1));
            data.wordsX = {source.words{pi.X}};
            data.GX     = GMCCA_backup.processGraph(source.G(pi.X,pi.X));
            
            data.Y      = target.features(pi.Y,:);
            data.Y(:,1) = log10(data.Y(:,1));
            data.wordsY = {target.words{pi.Y}};
            data.GY     = GMCCA_backup.processGraph(target.G(pi.Y,pi.Y));
            
            data.source = source;
            data.target = target;
            
            agg_pi=GMCCA_backup.find_matching(options, data);
            
            alignment = [data.wordsX', {data.wordsY{agg_pi}}']
        end
        
        function G = processGraph(G)
            G = (G + G')/2;
            g = sum(G,2);
            g(g==0)=1;
            g = 1./g;
            G = bsxfun(@times, G, g);
        end
        
        function sanityCheck()
            % create data
            N = 500; 
            D = 40;
            data_type = 0; % 0=mock data
            data = GMCCA_backup.loadMockData(data_type, N, D); 
            % create options
            T = 200; % at most 200 iterations
            d = 30;  % use 30 correlation dims
            weight_type = 'inner'; % inner product similarity
            options = GMCCA_backup.makeOptions(weight_type, T, d, 0,0); 
            agg_pi=GMCCA_backup.find_matching(options, data);
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
        
        function data = loadMockData(type, N, D)
            if type == 0                                %% Create mock data
                data.X = 2*randn(N, D);                 % generate random gaussian data
                noise = randn(N,D);
                norm(noise)
                data.Y = data.X + noise;                % Y = X + noise
                
                q = 0.9;
                true_pi = Util.randswap(1:N, q*N);     % permute Y
                data.Y = data.Y(true_pi, :);
                [randU,~] = svd(rand(D));
                data.Y = data.Y * randU;
                
                data.true_pi = true_pi;                 % figure out inverse permutation
                [sorted, sig]=sort(true_pi);
                data.inv_true_pi = sorted(sig);         % and save it.
                
                data.GX = zeros(N);                     % create empty graphs.
                data.GY = zeros(N);
            elseif type == 1 
                % mock data with graphs
            else
                error('unknown type %d\n', type);
            end
        end
    end
end

