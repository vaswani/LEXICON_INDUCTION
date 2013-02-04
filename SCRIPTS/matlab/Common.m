classdef Common

    methods(Static)
        
        function X = loadMat(F)
            filename = Common.getMatFilename(F);
            fprintf('Loading %s\n', filename);
            load(filename); % should contain X
        end
        
        function saveMat(F, X)
            filename = Common.getMatFilename(F);
            fprintf('Saving %s\n', filename);
            save(filename, 'X');
        end
        
        function filename = getMatFilename(F)
            filename = ['',F.features, '_', F.syns, '.mat'];
        end
        
        function P = getFreq(X)
            freq = X.features(:,1);
            [sorted, pi] = sort(freq, 'descend');
            for i=1:length(pi), % surprisingly fast for 10k
                P{i,1} = X.words{pi(i)};
                P{i,2} = sorted(i);
            end
        end
        
        function lookup=lookupNeighbors(X, word_id)
            % given word id, looks up its neighbors in the graph.
            lookup.word = X.id2word.get(word_id);
            row = X.G(word_id,:);
            lookup.id = word_id;
            j = 1;
            for ngbr_id = find(row), 
                lookup.ngbr_ids(j) = ngbr_id;
                lookup.ngbrs{j} = X.id2word.get(ngbr_id);
                j = j + 1;
            end
        end
    end
end

