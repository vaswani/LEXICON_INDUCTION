classdef BilexiconUtil
   
    methods(Static)
        %% FUNCTIONS TO LOAD THE BILEXICON
        function lex = new()
            lex.s2t = java.util.Hashtable; % source to target
            lex.t2s = java.util.Hashtable; % target to source
        end
        
        function lex = load(filename)
            load(filename); % loading B;
            lex = BilexiconUtil.parseLexicon(B);
        end
        
        function lex = parseLexicon(B)
            N = size(B,1);
            lex = BilexiconUtil.new();
            for n=1:N, % go over words.
                W1 = B{n,1};
                W2 = B{n,2};
                
                if length(W1)==0 || length(W2)==0 % skip empty values.
                    continue
                end
                % map source words to a list of target words and vice
                % versa.
    
                lex.s2t = BilexiconUtil.listadd(lex.s2t, W1, {W2});
                lex.t2s = BilexiconUtil.listadd(lex.t2s, W2, {W1});
            end
        end
        
        %% GENERAL HASH OF LIST FUNCTIONS
        
        function h = listadd(h, key, values)
            key = BilexiconUtil.fixWord(key);
            list = h.get(key);
            if isempty(list)
                h.put(key, java.util.LinkedList);
            end
            for i = 1:length(values),
                word = values{i};
                word = BilexiconUtil.fixWord(word);
                h.get(key).add(word);
            end
        end
        
        function word = fixWord(word)
            if ~strcmp(word(1), '_') % words are padded with starting and trailing '_'.
                word = ['_',word,'_'];
            end
        end

        function hash = new_addAll(list)
            % yes, I know there's a function addAll
            hash = java.util.HashSet;
            N = size(list,1);
            for n=1:N,
                hash.add(list{n});
            end
        end
        
        %% SET UP THE BILEXICON
        
        %function gtlex = ground_truth(lex, source_words, target_words, seed_source_words, seed_target_words)
        function gtlex = ground_truth(lex, source_words, target_words)
            % This function prepares the ground truth bilexicon - gtlex
            % scoring will be done against gtlex.
            % gtlex is a sub-set of lex, restricted to the given
            % source and target words.
            
            % setup a hash for quick lookup
            HS.source = BilexiconUtil.new_addAll(source_words);
            HS.target = BilexiconUtil.new_addAll(target_words);
    
            % remove words from the bilexicon lex.s2t/t2s if they don't
            % appear in the source/target.
            gtlex.s2t = BilexiconUtil.remove_missing_words(lex.s2t, HS.source, HS.target);
            gtlex.t2s = BilexiconUtil.remove_missing_words(lex.t2s, HS.target, HS.source);
        end

        function bilex = remove_existing_words(bilex, keys_to_remove)
            % removes a set of words from the hash
            keys_to_remove = keys_to_remove.toArray();
            for i=1:keys_to_remove.size(),
                key = keys_to_remove(i);
                if bilex.containsKey(key)
                    bilex.remove(key);
                end
            end
        end
        
        function new_bilex = remove_missing_words(bilex, sourceHS, targetHS)
            % In a clean hash, put only keys that are present in sourceHS
            % and, remove target words not in targetHS.
            % so that only possible edges can be counted
            source_words = sourceHS.toArray();
            new_bilex    = java.util.Hashtable;
            for i=1:source_words.size(),
                source_word = source_words(i);
                if bilex.containsKey(source_word)
                    target_list = bilex.get(source_word);
                    % remove words not in target
                    target_itr = target_list.listIterator();
                    while target_itr.hasNext()
                        target_word = target_itr.next();
                        if ~targetHS.contains(target_word)
                            target_itr.remove();
                        end
                    end
                    if target_list.size() > 0
                        new_bilex.put(source_word, target_list);
                    end
                end
            end
        end
        
        %% scoring
        
        function score = F1(precision, recall, beta)
            if nargin < 3
                beta = 1;
            end
            denom = beta^2*precision + recall;
            score = (1+beta^2).*(precision.*recall)./denom;
        end
        
        function scores = getF1scores(gtlex, matching, weights, options)
            % returns precision, recall and F1 scores.
            minWeight = min(weights);
            maxWeights = max(weights);
            %[matching, mat2cell(weights', ones(length(weights),1))]
            
            M = gtlex.s2t.size(); % we use the s2t mapping as the reference (practically s should be English, for which words are easy to obtain).
            % sort matching in ascending order.
            N = size(matching,1);
            assert(length(weights) == N);
            [weights,pi] = sort(weights);
            matching = matching(pi, :);

            C = zeros(N,3); 
            for n=1:N,
                source_word = matching{n,1};
                target_word = matching{n,2};
                C(n,1) = 1; % count size of matching being considered
                if gtlex.s2t.containsKey(source_word) % some match exists for this source word
                    C(n,2) = 1; % counts number of existing word in source and lexicon
                    C(n,3) = BilexiconUtil.is_valid_s2t_match(gtlex, source_word, target_word); % counts valid matches
                end
            end
            C = cumsum(C);
            
            % C1 - always 1, i.e. cumsum counts the number of words.
            % C2 - 1 of source
            % C1 - counts the size of the matching being considered
            
            scores.M = M;
            scores.precision = C(:,3) ./ C(:,2);
            scores.recall = C(:,3) ./ M;
            scores.F1 = BilexiconUtil.F1(scores.precision,scores.recall);
            scores.N = C(:,2);
            scores.R = C(:,1);
        end
        
        function b = is_valid_s2t_match(gtlex, source_word, target_word)
            b = 0;
            target_list = gtlex.s2t.get(source_word);
            if ~isempty(target_list)
                target_words = target_list.toArray();
                for i=1:target_list.size()
                    word = target_words(i);
                    if strcmpi(word, target_word)
                        b = 1;
                        break;
                    end   
                end
            end
        end
        
        %% OUTPUT
        
        function outputScores(scores, options, title)
            fprintf('Results for %s:\n', title);
            fprintf('===============\n');
            %options
            % find precisions based on recall levels
            i05 = find(scores.recall>=0.05,1);
            i10 = find(scores.recall>=0.1,1);
            i20 = find(scores.recall>=0.20,1);
            i25 = find(scores.recall>=0.25,1);
            i33 = find(scores.recall>=0.33,1);
            i40 = find(scores.recall>=0.40,1);
            i50 = find(scores.recall>=0.50,1);
            i60 = find(scores.recall>=0.60,1);
            
            I = unique([i05, i10, i20, i25, i33, i40, i50, i60]);
            
            N  = scores.R(I);
            P  = scores.precision(I);
            R  = scores.recall(I);
            F1 = scores.F1(I);
            fprintf('Ra'); fprintf('%8.2f',100*N/max(scores.R)); fprintf('\n');
            fprintf('F1'); fprintf('%8.2f', 100*F1); fprintf('\n');
            fprintf('Pr'); fprintf('%8.2f', 100*P);  fprintf('\n');
            fprintf('Re'); fprintf('%8.2f', 100*R);  fprintf('\n');
            fprintf('=======\n');
            if ~strcmpi(title, 'Edit Distance')
                fprintf('Params: [wtype="%s", lambda=%2.2f, M=%d, K=%2.2f]\n', options.weight_type, options.lambda, options.M, options.K);
                fprintf('=======\n');
            end
        end
        
        %%%%%%%%%%%%%%%%% TEST FUNCTIONS
           
        
        function gtlex = test()
            % test function for this class.
            B = {
                '_big_', '_grande_';
                '_the_', '_el_';
                '_the_', '_la_';
                '_the_', '_un_';
                '_thin_', '_delgado0_';
                '_thin_', '_delgado1_';
                '_city_', '_ciudad_';
                '_country_', '_pais_';
                '_beautiful_', '_hermosa_';
                };
            
            % build the lexicon from B.
            bilex = BilexiconUtil.parseLexicon(B);
            
            matching = {
                '_big_', '_delgado1_';
                '_city_', '_el_';
                '_thin_', '_delgado0_';
                '_country_', '_hermosa_';
                '_beautiful_', '_pais_';
                '_unmatched_', '_unmatched_'
                };
            
            weights = 1:size(matching,1); % just some simple weights
            
            % filter with ground truth.
            gtlex  = BilexiconUtil.ground_truth(bilex, matching(:,1), matching(:,2));
            scores = BilexiconUtil.getF1scores(gtlex, matching, weights);
            plot(scores.F1);
            
            BilexiconUtil.outputScores(scores, [], 'test run');
        end
    end
    
end

