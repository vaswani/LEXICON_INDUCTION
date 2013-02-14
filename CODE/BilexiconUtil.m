classdef BilexiconUtil
   
    methods(Static)
        
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
        
        function gtlex = ground_truth(lex, source_words, target_words)
            % setup a hash for quick lookup
            N.source = size(source_words,1);
            N.target = size(target_words,1);
            HS.source = java.util.HashSet;
            HS.target = java.util.HashSet;
            for n=1:N.source,
                HS.source.add(source_words{n});
            end
            for n=1:N.target
                HS.target.add(target_words{n});            
            end
                            
            % remove words from the lexicon if they don't appear in A.
            gtlex.s2t = BilexiconUtil.remove_missing_words(lex.s2t, HS.source, HS.target);
            gtlex.t2s = BilexiconUtil.remove_missing_words(lex.t2s, HS.target, HS.source);
        end
        
        function new_bilex = remove_missing_words(bilex, sourceHS, targetHS)
            % put in clean hash, only keys that are present in sourceHS
            % moreover, remove target words not in targetHS.
            source_words = sourceHS.toArray();
            new_bilex = java.util.Hashtable;
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
        
        function F1 = scoreAlignment(lex, A, coeff)
            N = size(A,1);
            count = 0;
            for n=1:N,
                source_word = A{n,1};
                target_word = A{n,2};
                list2 = lex.s2t.get(source_word);
                for i = 0:list2.size()-1,
                    word = list2.get(i);
                    if strcmpi(word, target_word)
                        count = count + 1;
                        break;
                    end
                end
            end
        end
        
        function score = F1(precision, recall, beta)
            if nargin < 3
                beta = 1;
            end
            denom = beta^2*precision + recall;
            score = (1+beta^2)*(precision*recall)/denom;
        end
        
        function scores = getF1scores(gtlex, matching, weights)
            % returns precision, recall and F1 scores.
            minWeight = min(weights);
            maxWeights = max(weights);
            
            K = 5;
            range = linspace(minWeight,maxWeights,K);
            M = gtlex.s2t.size(); % we use the s2t mapping as the reference (practically s should be English, for which words are easy to obtain).
            for r = 1:length(range)-1,
                I = weights<=range(r+1);
                sub_matching = matching(I,:);
                
                tp = 0; % will count the edges in gtlex that exist in the matching
                Nr = 0; % number of proposed matches that can actually be matched.
                for n=1:size(sub_matching,1),
                    source_word = sub_matching{n,1};
                    target_word = sub_matching{n,2};
                    if gtlex.s2t.containsKey(source_word) % some match exists for this source word
                        Nr = Nr + 1;
                        % do we have a correct match?
                        if BilexiconUtil.is_valid_s2t_match(gtlex, source_word, target_word)
                            tp = tp + 1; % yes we do!
                        else
                            tp = tp;
                        end
                    end
                end
                scores.N(r) = Nr;
                scores.precision(r) = tp/Nr;    % Nr=(tp+fp)=# of retrieved matchings
                scores.recall(r)    = tp/M;     % M=(tp+fn) =# of relevant matchings.
                scores.F1(r) = BilexiconUtil.F1(scores.precision(r), scores.recall(r));
            end
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
        
%         function score = F1v2(tp, fn, fp, beta)
%             num = (1+beta^2)*tp;
%             denom = num + beta^2*fn + fp;
%             score = num / denom;
%         end
            
        
        function gtlex = test()
            %% create a bilexicon
            % A1 -- B1
            % A2 -- B2_0, B2_1, C
            % A3 -- B3_0,D
            % A4 -- B2_0
            
            B = {
                'big', 'grande';
                'the', 'el';
                'the', 'la';
                'the', 'un';
                'thin', 'delgado0';
                'thin', 'delgado1';
                'city', 'ciudad';
                'country', 'pais';
                'beautiful', 'hermosa';
                }
            bilex = BilexiconUtil.parseLexicon(B);
            
            matching = {
                'big', 'delgado1';
                'city', 'el';
                'thin', 'delgado0';
                'country', 'hermosa';
                'beautiful', 'pais';
                'unmatched', 'unmatched'
                }
            
            weights = [1,2,3,4,5];
            
            gtlex = BilexiconUtil.ground_truth(bilex, matching(:,1), matching(:,2));
            scores = BilexiconUtil.getF1scores(gtlex, matching, weights);
            plot(scores.F1);
        end
    end
    
end

