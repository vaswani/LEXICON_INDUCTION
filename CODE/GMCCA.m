classdef GMCCA
    
    methods(Static)
        
        function F=find_matching(options, data)
            % computes the within graph aided mCCA between X and Y.
            % returns pi, the permutation that matches samples (rows) of X
            % with rows of Y.
            [N.X,DX] = size(data.X.features);
            [N.Y,DY] = size(data.Y.features);
            [N.rX,~] = size(data.X.rest.features);
            [N.rY,~] = size(data.Y.rest.features);
            [N.fX,~] = size(data.X.fixed.features);
            [N.fY,~] = size(data.Y.fixed.features);
            assert(N.X==size(data.X.G,1));
            assert(N.Y==size(data.Y.G,1));
            assert(N.rX==N.rY);
            assert(N.fX==N.fY);
            %%
            featuresX = [data.X.rest.features; data.X.fixed.features]; % this should == data.X.features
            start = tic; % measure start time
            F.hamming = inf(options.T, 1);
            F.weights = [inf(1,N.rX),-inf(1,N.fX)];
            seed_pi = [(N.X-N.fX+1):N.X]; % this part of the permutation stays fixes (at the end)
            pi_t = 1:N.rY;                % this part of the permutation changes (at the beginning)
            restID = 1:N.rY;
            pm.start_length = length(seed_pi);
            pm.length = pm.start_length % pm = partial matching, initial length and increment steps
            pm.i = 1;
            for t=1:options.T,
                options.t = t;
                inv_pi = Util.inverse_perm(pi_t);
                featuresY = [data.Y.rest.features(pi_t,:); data.Y.fixed.features];
                % supposedly, here, featuresX is aligned with featuresY, 
                % however, here we take the t/10 lowest scoring pairs for the
                % sake of computing the new latent representation.
                sorted_weights = sort(F.weights); 
                
                top_matches = find(F.weights <= sorted_weights(pm.length));
                % compute latent representation model under partial matching
                cca_model = CCAUtil.latentCCA(featuresX(top_matches,:), featuresY(top_matches,:), options);  
                % using model, compute latent representation of entire matching
                Z = CCAUtil.getLatent(cca_model, featuresX, featuresY);
                
                Z.Y    = Z.Y([inv_pi,seed_pi],:);                                                % permute back.
                W_t    = MatchingUtil.makeWeights(options, Z.X, Z.Y, data.X.G, data.Y.G);       % compute weights 
                
                assert(size(W_t,1) == size(W_t,2));
                old_pi = pi_t;
                %figure; imagesc(W_t(:, pi_t));
                [pi_t, F.cost(t), F.edge_cost{t}] = MatchingUtil.match(W_t(restID,restID), options);                      % compute matching
                % pi_t = Util.randswap(pi_t, 4);
                Util.is_perm(pi_t); %% assert pi_t is a valid permutation
                F.pi = [pi_t, seed_pi];
                F.weights = [F.edge_cost{end}, -inf(1, N.fX)];
                GMCCA.getMatching(data.X.words, data.Y.words, F);
                % log and output
                %F.normXY(t) = norm(Z.X-Z.Y,'fro');
                %F.normX(t)  = norm(Z.X, 'fro');
                %F.normY(t)  = norm(Z.Y, 'fro');
                F.inner(t)  = sum(sum(Z.Y .* Z.X));
                if isfield(data, 'true_pi') % compute distance to true permutation.
                    F.hamming(t) = Util.hamming(Util.inverse_perm(pi_t), data.true_pi_seedless); 
                    fprintf('%d), cost=%2.3f  inner=%2.3f  hamming=%f\n',t, F.cost(t), F.inner(t), F.hamming(t));
                else
                    F.hamming(t) = Util.hamming(pi_t, old_pi);
                    fprintf('%d), cost=%2.3f  inner=%2.3f  hamming_change=%d\n',t, F.cost(t), F.inner(t), F.hamming(t));
                end
                
                if F.hamming(t)==0 && pm.length == N.X 
                    % stopping condition is (1) fixed permutation and (2) no more edges to add
                    fprintf('Fixed point after t=%d iterations\n', t);
                    break;
                else
                    % increase the partial matching length
                    pm.length = min(floor(pm.start_length + N.X*pm.i*options.delta_pm), N.X);
                    pm.i = pm.i + 1; % count increment.
                    fprintf('increased pm.length=%d\n',pm.length);
                end
            end

            F.end = toc(start);
            F.pi = [pi_t, seed_pi];                                     % augment the seed to the end
            F.weights = [F.edge_cost{end},-ones(1, length(seed_pi))];     % and add inf weights as well
        end
        
        function run(exp_id, maxN, lambda, M, K, weight_type)
            % exp_id - experiment id, used in output filenames.
            % maxN - the maximum number of samples to consider.
            
            %% OPTIONS            
            fprintf('------------- Starting -----------\n');
            %weight_type = 'dist'; % 'inner or 'dist'
            
            T = 30;  % at most 30 iterations
            delta_pm = 0.05;
            max_seed = 300;
            d = 0;
            if nargin < 3
                lambda = 0; % diffusion rate
                M = 1;   % random walk steps
                K = 20; % number of neighbors
                
            end
            
            options = GMCCA.makeOptions(weight_type, T, d, M,lambda,K, max_seed, delta_pm); 
            
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
            match = MatchingUtil.init_matching(data.X.words, data.Y.words, 1, options.max_seed);
            data.seed.match = match;
            data.seed.N = length(match.source);
            results.edit_distance = [data.X.words(match.all.source), data.Y.words(match.all.target), (mat2cell(match.all.weights', ones(maxN,1)))];
            Common.outputCSV(exp_id,'edit_distance', results.edit_distance);
            
            %% evaluate edit_distance matching
            lex   = BilexiconUtil.load(lexicon.filename);
            source_no_seed = setdiff(1:maxN,match.source);
            target_no_seed = setdiff(1:maxN,match.target);
            gtlex = BilexiconUtil.ground_truth(lex, data.X.words(source_no_seed), data.Y.words(target_no_seed));
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
            matching.mcca = GMCCA.getMatching(data.X.words, data.Y.words, F);
            count_stationary = sum(strcmpi(data.Y.words,  data.Y.words(F.pi)));
            scores.mcca = BilexiconUtil.getF1scores(gtlex, matching.mcca(:,2:3), cell2mat(matching.mcca(:,4)));
            BilexiconUtil.outputScores(scores.mcca, options, 'MCCA');
            %plot(F.inner);
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
            
            feature_sum = sum(X.features > 0);
            frequent    = feature_sum >  40; % find features that appear more than X times
%              sparse10    = feature_sum >= 0 & feature_sum <= 10; % find features that appear more than X times
%              sparse20    = feature_sum > 10 & feature_sum <= 20; % find features that appear more than X times
%              sparse30    = feature_sum > 20 & feature_sum <= 30; % find features that appear more than X times
%              sparse40    = feature_sum > 30 & feature_sum <= 40; % find features that appear more than X times
%              X.features  = [X.features(:, frequent), ...
%                             sum(X.features(:,sparse10),2)...
%                             sum(X.features(:,sparse20),2)...
%                             sum(X.features(:,sparse30),2)...
%                             sum(X.features(:,sparse40),2)]; % take frequent features and collapse rare
%             
            sparse40    = feature_sum <= 40;
            X.features  = [X.features(:, frequent), sum(X.features(:,sparse40),2)];
            
            
            %H = Util.knngraph(X.features, options.K+1);
%           H = Util.epsgraph(X.features, options.K);
            %H = H - eye(size(H,1)); % remove self as neighbor.
            %fprintf('Using %d edges in graph\n', sum(H(:)));
            %X.G     = Util.to_stochastic_graph(H);
            X.G = source.G;
            %% add log frequency and log length (but don't use them in the graph)
            X.features  = [logFr, log2(L), X.features];
%            X.features = [X.features, X.G*X.features];
            [N2,D2] = size(X.features);
            fprintf('Setup features from [%d,%d] to [%d,%d].\n', N1,D1, N2,D2);
        end
        
        function matching = getMatching(wordsX, wordsY, F)
            N = length(wordsX);
            indices = (mat2cell([1:N]', ones(N,1)));
            if ~isfield(F, 'weights')
                F.weights = F.pi;
            end
            F.weights = Util.ascol(F.weights);
            weights = (mat2cell(F.weights, ones(N,1)));
            matching = [ indices,wordsX, wordsY(F.pi), weights];
            [~, sigma] = sort(F.weights, 'descend');
            matching = matching(sigma, :);
        end
        
        function X = fix_matched_words(X, seed_pi)
            % move seed matched words to the end.
            Nw = length(X.words);
            rest = setdiff(1:Nw, seed_pi);
            new_order = [rest, seed_pi];
            
            %% store seed-matched words in 'fixed' and the rest in 'rest'.
            X.fixed.words    = X.words(seed_pi);
            x.rest.word      = X.words(rest);
            X.fixed.features = X.features(seed_pi,:);
            X.rest.features  = X.features(rest,:);

            X.words = X.words(new_order);
            X.features = X.features(new_order,:);
            X.G = X.G(new_order,new_order);
            
            featuresX = [X.rest.features; X.fixed.features];
            assert(norm(featuresX - X.features, 'fro')==0);
        end
        
        function hamdist = sanityCheck(seed, data_noise, graph_noise, lambda_coeff, M, K)
            if nargin == 0
                seed = 3;
                data_noise   = 0.7;
                graph_noise  = 0;
                lambda_coeff = 1;
                K = 20;
                M = 1;
                max_seed = 100;
                delta_pm = 0.05;
            end
            rng(seed);
            % create data
            N = 1000; 
            D = 100;
            data_type = 1; % 0=mock data with empty graph, 1=mock data with sparse graph
            data = GMCCA.loadMockData(data_type, N, D, data_noise, graph_noise); 
            % create options
            T = 20; % at most 200 iterations
            d = D;  % use 30 correlation dims
            weight_type = 'inner'; % inner product similarity
            if data_noise == 0
                lambda = 1;
            else
                lambda = lambda_coeff * data_noise;
            end
            options = GMCCA.makeOptions(weight_type, T, d, M, lambda, K, max_seed, delta_pm); 
            F=GMCCA.find_matching(options, data);
            
            alignment = GMCCA.getMatching(data.X.words, data.Y.words, F);
            
            recovered = all([data.X.words{:}] == [data.Y.words{F.pi}]);
            hamdist = Util.hamming(Util.inverse_perm(F.pi), data.true_pi);
            fprintf('Recovered = %d hamdist=%d\n', recovered, hamdist);
        end
        
        function plot_prob_of_recovery()
            % these settings can provide a nice graph.
            noise_coeffs = 0:0.1:0.7;
            N_NC = length(noise_coeffs);
            
            T = 10;
            K = 5;
            lambda_coeff = 5;
            tic;
            hamdist.base  = zeros(N_NC,T);
            hamdist.graph = zeros(N_NC,T);
            for i = 1:N_NC,
                nc = noise_coeffs(i);
                for t = 1:T,
                    randseed = t;
                    hamdist.base(i,t)  = GMCCA.sanityCheck(randseed, nc, nc, lambda_coeff, 0, 0);
                    hamdist.graph(i,t) = GMCCA.sanityCheck(randseed, nc, nc, lambda_coeff, 1, K);
                    fprintf('[%d %d] / [%d %d]\n', i,t,N_NC, T);
                end
            end
            hamdist.base
            hamdist.graph
            plot([mean(hamdist.base<=100,2), mean(hamdist.graph<=100,2)]);
            legend({'baseline', 'graph'});
            toc;
        end
    end
    
    methods(Static, Access=private)
        
        function options = makeOptions(weight_type, T, d, M, lambda, K, max_seed, delta_pm)
            options.T = T;                         % MAX "EM" ITERATIONS
            options.weight_type = weight_type;     % commput 'inner' or 'dist' weight.
            options.d = d;                         % ignore this.
            options.K = K;                         % parameter for graph (for knn-graphs, K is the number of neighbors, eps neighborhood graphs - eps.)
            options.lambda = lambda;               % weight assigned to graph features and
            options.M = M;                         % degree to use graph.
            % for example, K+(lambda*G)K+(lambda*G)^2K .. *(lambda*G)^MK,
            % try M=1, lambda>1 first.
            options.max_seed = max_seed;           % maximum length of seed.
            options.delta_pm = delta_pm;           % increment percent of partial matching 
        end
        
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
                k = 3;
                data.X.G = Util.knngraph(Z,k);
                %data.X.G = rand(N,N)<0.1;
            else
                error('unknown type %d\n', type);
            end
            
            data.X.features = Z + data_noise*randn(N,D);                % X = Z + noise
            data.Y.features = Z + data_noise*randn(N,D);                % Y = Z + noise
            q = 0.92;
            qN = floor(q*N);
            seed = (qN+1):N;
            true_pi         = [Util.randswap(1:qN, qN), seed];     % permute Y
            
            data.Y.features = data.Y.features(true_pi, :);
            data.Y.words    = data.X.words(true_pi);
            
            data.Y.G = data.X.G(true_pi, true_pi);
            data.X.G = and(data.X.G, (rand(N,N)>graph_noise)); % remove a few randomly.
            data.Y.G = and(data.Y.G, (rand(N,N)>graph_noise));
            data.X.G = Util.to_stochastic_graph(data.X.G);
            data.Y.G = Util.to_stochastic_graph(data.Y.G);
            
            data.X.features = GMCCA.random_rotate_shift(data.X.features);
            data.Y.features = GMCCA.random_rotate_shift(data.Y.features);
            % fix the seed to the beginning
            [data.X] = GMCCA.fix_matched_words(data.X, seed);
            [data.Y] = GMCCA.fix_matched_words(data.Y, seed);
            
            % F.pi = 1:N; 
            % matching = GMCCA.getMatching(data.X.words, data.Y.words, F); % view the matching; % view the matching
            % F.pi = Util.inverse_perm(true_pi);
            % matching = GMCCA.getMatching(data.X.words, data.Y.words, F)
            data.true_pi = true_pi;
            data.true_pi_seedless = true_pi(1:qN);  % figure out inverse permutation
        end
    end
end