function lexsyn2mat(F)
    % input:
    % * featfile - filename containing a list of words with their features (denoted X)
    % * synfile  - filename containing a list of pairs of words (synonyms) E
    % 
    % output:
    % * saves a file
    import java.util.Hashtable;
    X   = read_features(F.features);
    X.G = make_synonym_adj(F.syns, X);
    X.F = F;
    X.created  = datestr(now);
    Common.saveMat(F, X);
end

function G = make_synonym_adj(filename, X)
    G   = sparse(zeros(X.N));
    fid = fopen([filename,'.txt']);
    pairs_skipped = 0;
    pair_count    = 0;
    while 1 % read lines one by one.
        line = fgetl(fid);
        if ~ischar(line) 
            break 
        end
        pair_count = pair_count + 1;
        %pat = '([\S]+),(\S+)'; % the pattern to extract.
        %matches = regexp(line, pat ,'tokens');
        matches = split(line, ',');
        word1      = matches{1};
        word2      = matches{2};
        weight     = matches{3};
        id1 = X.word2id.get(word1);
        id2 = X.word2id.get(word2);
        if isempty(id1) || isempty(id2)
            if isempty(id1)
                fprintf('skipping unknown word1="%s"\n',word1);
            end
            if isempty(id2)
                fprintf('skipping unknown word2="%s"\n',word2);
            end
            pairs_skipped = pairs_skipped + 1;
        else
            if G(id1,id2)
                fprintf('words already exist "%s"=%d "%s"=%d.\n', word1, id1, word2, id2);
            end
            G(id1,id2) = weight;
        end
    end
    fprintf('%d / %d pairs skipped.\n', pairs_skipped, pair_count);
    fclose(fid);
end

function X=read_features(filename)
    % read the file
    Y = importdata([filename,'.txt']);
    X.features = Y.data;
    X.words = Y.textdata;
    N = length(X.words);
    
    X.word2id = java.util.Hashtable;
    X.word2id = java.util.Hashtable;
    X.id2word = java.util.Hashtable;
    
    for n=1:N, % supplied by importdata in our case
        word    = X.words{n};
        X.word2id.put(word,n);
        X.id2word.put(n,word);
    end
    X.N = N;
end
