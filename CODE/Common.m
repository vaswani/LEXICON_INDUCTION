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
        
        function lookup=lookupNeighbors(X, word_id, G)
            % given word id, looks up its neighbors in the graph.
            lookup.word = X.id2word.get(word_id);
            row = G(word_id,:);
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
        
        function [hashA, hashB, N] = getFilenameHashes(listA, listB)
            % make a filename basedon the lists.
            hashA = Common.str2hash([listA{:}]); % hash the list to a number
            hashB = Common.str2hash([listB{:}]);
            N = length(listA);
        end
        
        function filename = to_pcamodel_filename(list)
            [hashA, ~, N] = Common.getFilenameHashes(list, list);
            filename = ['cache/pca_A=',num2str(hashA), '_N=',num2str(N), '.mat'];
        end
        
        function filename = to_editdist_filename(listA, listB)
            [hashA, hashB, N] = Common.getFilenameHashes(listA, listB);
            filename = ['cache/ed_A=',num2str(hashA), '_B=', num2str(hashB), '_N=',num2str(N), '.mat'];
        end
        
        function W = load_editdist_file(listA, listB)
            filename = Common.to_editdist_filename(listA, listB);
            % load if exists.
            if exist(filename, 'file')
                fprintf('Loaded edit distance file "%s".\n', filename);
                load(filename);
            else
                fprintf('Edit distance file "%s" not found.\n', filename);
                W = [];
            end
        end
        
        function X = cachePCA(words, X)
            filename = Common.to_pcamodel_filename(words);
            if exist(filename, 'file')
                fprintf('Loading existing PCA model file "%s"', filename);
                load(filename);
            else
                model = PCAUtil.PCA(X);
                save(filename, 'model');
                fprintf('Saved  PCA model file "%s"', filename);
            end
            d = PCAUtil.getEigenmassDim(model, 0.9); % d top dimensions that capture 90% of the variance in the data
            X = PCAUtil.project(model, X, d);
        end
        
        function save_editdist_file(listA, listB, W)
            filename = Common.to_editdist_filename(listA, listB);
            save(filename, 'W');
            fprintf('Saved edit distance file "%s".\n', filename);
        end
        
        function outputCSV(exp_id, mtype, A)
            filename = sprintf('output/matching_exp_id=%d_type=%s.txt', exp_id, mtype);
            fid = fopen(filename,'w');
            [N,D] = size(A);
            for n=1:N,
                str = '';
                for d=1:D,
                    v = A{n,d};
                    if ~ischar(v)
                        v = num2str(v);
                    end
                    str = [str, ',', v];
                end
                str = str(2:end);
                str = regexprep(str, '%', '%%');
                fprintf(fid,[str, '\n']);
            end
            
            fclose(fid);
        end
        
        function A=loadSeed(filename)
            fid = fopen(filename);
            j = 1;
            while 1 % read lines one by one.
                line = fgetl(fid);
                if ~ischar(line) 
                    break 
                end
                pat = '(\S+)\s+(\S+)'; % the pattern to extract.
                matches = regexp(line, pat ,'tokens');
                matches = matches{1};
                word1      = matches{1};
                word2      = matches{2};
                A{j,1} = word1;
                A{j,2} = word2;
                j = j + 1;
            end
            fclose(fid);
        end
    end
end

