classdef Common

    methods(Static)
        
        function X = loadMat(S)
            fprintf('Loading %s\n', S.filename);
            load(S.filename); % should contain X
        end
        
        
        function [P, pi] = getFreq(X)
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
        
        function hash = str2hash(str)
            % From string to double array
            str=double(str);
            if(nargin<2), type='djb2'; end
            switch(type)
                case 'djb2'
                    hash = 5381*ones(size(str,1),1); 
                    for i=1:size(str,2), 
                        hash = mod(hash * 33 + str(:,i), 2^32-1); 
                    end
                case 'sdbm'
                    hash = zeros(size(str,1),1);
                    for i=1:size(str,2), 
                        hash = mod(hash * 65599 + str(:,i), 2^32-1);
                    end
                otherwise
                    error('string_hash:inputs','unknown type');
            end
        end
        
        function filename = to_editdist_filename(listA, listB, max_dist)
            % make a filename basedon the lists.
            hashA = Common.str2hash([listA{:}]); % hash the list to a number
            hashB = Common.str2hash([listB{:}]);
            N = length(listA);
            filename = ['cache/ed_A=',num2str(hashA), '_B=', num2str(hashB), '_N=',num2str(N), '_maxdist=',num2str(max_dist), '.mat'];
        end
        
        function W = load_editdist_file(listA, listB, max_dist)
            filename = Common.to_editdist_filename(listA, listB, max_dist);
            % load if exists.
            if exist(filename, 'file')
                fprintf('Loaded edit distance file "%s".\n', filename);
                load(filename);
            else
                fprintf('Edit distance file "%s" not found.\n', filename);
                W = [];
            end
        end
        
        function save_editdist_file(listA, listB, max_dist, W)
            filename = Common.to_editdist_filename(listA, listB, max_dist);
            save(filename, 'W');
            fprintf('Saved edit distance file "%s".\n', filename);
        end
    end
end

