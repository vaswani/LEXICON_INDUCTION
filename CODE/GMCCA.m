classdef GMCCA
    
    methods(Static)
        
        function F=find_matching(options, data)
            % computes the within graph aided mCCA between X and Y.
            % returns pi, the permutation that matches samples (rows) of X
            % with rows of Y.
            start = tic();
            if options.use_single == 1
                data.X.features = single(data.X.features); % work w/ single precision.
                data.Y.features = single(data.Y.features);
            end
            [NX,DX] = size(data.X.features);
            [NY,DY] = size(data.Y.features);
            assert(NX==size(data.X.G,1));
            assert(NY==size(data.Y.G,1));
            start = tic;
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
                %figure; imagesc(W_t(:, pi_t));
                [pi_t, F.cost(t), F.edge_cost{t}] = MatchingUtil.match(W_t, options);                      % compute matching
                % pi_t   = Util.randswap(pi_t, 4);
                Util.is_perm(pi_t); %% assert pi_t is a valid permutation
                
                % log and output
                F.normXY(t) = norm(Z.X-Z.Y,'fro')^2;
                if isfield(data, 'true_pi')   % compute distance to true permutation.
                    F.hamming(t) = Util.hamming(Util.inverse_perm(pi_t), data.true_pi); 
                    fprintf('%d), cost=%2.3f\thamming=%f\n',t, F.cost(t), F.hamming(t));
                else
                    F.hamming(t) = Util.hamming(pi_t, old_pi);
                    fprintf('%d), cost=%2.3f\thamming_change=%d\n',t, F.cost(t), F.hamming(t));
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
            F.end = toc(start);
            F.pi = pi_t;
            F.weights = F.edge_cost{end};
        end
        
        function options = makeOptions(weight_type, T, d, K, lambda)
            options.T = T;
            options.weight_type = weight_type;
            options.d = d;
            options.K = K;
            options.lambda = lambda;
            options.use_single = 0;
        end
        
        function run(exp_id, maxN)
            % exp_id - experiment id, used in output filenames.
            % maxN - the maximum number of samples to consider.
            
            %% OPTIONS            
            fprintf('------------- Starting -----------\n');
            weight_type = 'dist'; % 'inner or 'dist'
            
            T = 20;  % at most 20 iterations
            K = 1;   % random walk steps
            lambda = 0.25; % diffusion rate
            d = 0;
            options = GMCCA.makeOptions(weight_type, T, d, K,lambda); 
            
            %% DATA
            %source.filename = './data/en.ortho.v1_en.syns.v1.mat';
            %target.filename = './data/es.ortho.v1_es.syns.v1.mat';
%             source.filename = './data/FEB3_en.features.10k_en.syns.v1.mat';
%             target.filename = './data/FEB3_es.features.10k_es.syns.v1.mat';
            source.filename = './data/FEB6_en.features_space.10k_en.syns.v2.mat';
            target.filename = './data/FEB6_es.features_space.10k_es.syns.v2.mat';
            lexicon.filename = 'data/wiktionary_bilexicon_en-es.mat'; 

            source = Common.loadMat(source);
            target = Common.loadMat(target);
            [~, source.pi] = Common.getFreq(source);
            [~, target.pi] = Common.getFreq(target);
            
            data.X = GMCCA.setup_features(options, source, maxN);
            data.Y = GMCCA.setup_features(options, target, maxN);
            
            % figure out initial matching.
            % for now, based on edit-distance
            match = MatchingUtil.init_matching(data.X.words, data.Y.words, 1);
            data.seed.match = match;
            data.seed.N = length(match.source);
            results.edit_distance = [data.X.words(match.all.source), data.Y.words(match.all.target), (mat2cell(match.all.weights', ones(maxN,1)))];
            Common.outputCSV(exp_id,'edit_distance', results.edit_distance);
            
            %% evaluate edit_distance matching
            lex   = BilexiconUtil.load(lexicon.filename);
            gtlex = BilexiconUtil.ground_truth(lex, data.X.words, data.Y.words);
            matching.edit_dist = GMCCA.getMatching(data.X.words(match.all.source), data.Y.words(match.all.target), match.all);
            scores.edit_dist   = BilexiconUtil.getF1scores(gtlex, matching.edit_dist(:,2:3), cell2mat(matching.edit_dist(:,4)));
            BilexiconUtil.outputScores(scores.edit_dist, options, 'Edit Distance');
            
            [data.X] = GMCCA.fix_matched_words(data.X, match.source);
            [data.Y] = GMCCA.fix_matched_words(data.Y, match.target);
            %init_alignment = GMCCA.getAlignment(data.X.words, data.Y.words)
            
            data.source = source;
            data.target = target;
            
            p = 1; % start with d that preserves at least p of the eigenmass of X and Y
            options.d = min(Util.mass_to_dim(data.X.features, p), Util.mass_to_dim(data.Y.features, p)); % use d correlation dims  
            
            fprintf('rank(X)=%d, rank(Y)=%d\n',rank(data.X.features), rank(data.Y.features));
            F = GMCCA.find_matching(options, data);
            
            %% output
            matching.mcca = GMCCA.getMatching(data.X.words, data.Y.words, F)
            scores.mcca = BilexiconUtil.getF1scores(gtlex, matching.mcca(:,2:3), cell2mat(matching.mcca(:,4)));
            BilexiconUtil.outputScores(scores.mcca, options, 'MCCA');
            %plot(F.normXY);
            fprintf('time: %2.2f\n', F.end);
            %match
        end
        
        function X = setup_features(options, source, maxN)
            % pick the top maxN most frequent words
            [N1,D1] = size(source.features);
            source.pi   = source.pi(1:maxN);
            pi          = source.pi;
            X.features  = source.features(pi,:); % sort words by frequency
            X.words     = source.words(pi);
            
            logFr       = log2(X.features(:,1)); % replce frequency to log2(freq)
            L           = Util.strlen(X.words);
            X.features(:,1) = [];
            
            frequent    = sum(X.features > 0) >  40; % find features that appear more than X times
            sparse      = sum(X.features > 0) <= 40; % find features that appear more than X times
            X.features  = [X.features(:, frequent), sum(X.features(:,sparse),2)]; % take frequent features and collapse rare
            % normalize features
            %V = sqrt(sum(X.features.^2,2));
            %X.features = bsxfun(@rdivide, X.features, V);
            
            %X.features  = [logFr, log2(L), X.features];
            X.G         = Util.to_stochastic_graph(source.G(pi, pi));
            [N2,D2] = size(X.features);
            
            fprintf('Setup features from [%d,%d] to [%d,%d].\n', N1,D1, N2,D2);
        end
        
        function matching = getMatching(wordsX, wordsY, F)
            N = length(wordsX);
            indices = (mat2cell([1:N]', ones(N,1)));
            weights = (mat2cell(F.weights', ones(N,1)));
            matching = [ indices,wordsX, wordsY(F.pi), weights];
        end
        
        function X = fix_matched_words(X, pi)
            % move seed matched words to the beginning.
            Nw = length(X.words);
            
            rest = setdiff(1:Nw, pi);
            new_order = [pi, rest];
            
            X.words = X.words(new_order);
            X.features = X.features(new_order,:);
            X.G = X.G(new_order,new_order);
        end
        
        function recovered = sanityCheck(seed, data_noise, graph_noise, lambda_coeff, K)
            if nargin == 0
                seed = 3;
                data_noise = 0.2;
                graph_noise = 0.2;
                lambda_coeff = 1;
                K = 0;
            end
            rng(seed);
            % create data
            N = 2000; 
            D = 200;
            data_type = 1; % 0=mock data with empty graph, 1=mock data with sparse graph
            data = GMCCA.loadMockData(data_type, N, D, data_noise, graph_noise); 
            % create options
            T = 20; % at most 200 iterations
            d = D;  % use 30 correlation dims
            weight_type = 'inner'; % inner product similarity
            lambda = lambda_coeff / data_noise;
            options = GMCCA.makeOptions(weight_type, T, d, K, lambda); 
            F=GMCCA.find_matching(options, data);
            alignment = GMCCA.getMatching(data.X.words, data.Y.words, F)
            
            recovered = all([data.X.words{:}] == [data.Y.words{F.pi}]);
            fprintf('Recovered = %d hamdist=%d\n', recovered, Util.hamming(Util.inverse_perm(F.pi), data.true_pi));
            %F.normXY
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
            rand_mean = 10*rand(1,D);
            F = F * rand_rot;
            F = bsxfun(@plus, F, rand_mean); % shift randomly.
        end
        
        function data = loadMockData(type, N, D, data_noise, graph_noise)
            data.X.words = mat2cell(1:N,1,ones(N,1))';
            Z = randn(N, D);         % generate random gaussian data
            if type == 0                             %% Create mock data    
                data.X.G = zeros(N);                 % create empty graphs.
            elseif type == 1 
                % mock data with graphs
                knnclassify(Z,Z,3);
                
                data.X.G = rand(N,N)<0.1;
            else
                error('unknown type %d\n', type);
            end
            
            data.X.features = Z + data_noise*randn(N,D);                % Y = X + noise
            data.Y.features = Z + data_noise*randn(N,D);                % Y = X + noise
            q = 0.92;
            true_pi         = Util.randswap(1:N, q*N);     % permute Y
            data.Y.features = data.Y.features(true_pi, :);
            data.Y.words    = data.X.words(true_pi);
            
            data.Y.G = data.X.G(true_pi, true_pi);
            data.X.G = and(data.X.G, (rand(N,N)>graph_noise)); % remove a few randomly.
            data.Y.G = and(data.Y.G, (rand(N,N)>graph_noise));
            data.X.G = Util.to_stochastic_graph(data.X.G);
            data.Y.G = Util.to_stochastic_graph(data.Y.G);
            
            data.X.features = GMCCA.random_rotate_shift(data.X.features);
            data.Y.features = GMCCA.random_rotate_shift(data.Y.features);

            data.true_pi = true_pi;                 % figure out inverse permutation
        end
    end
end

