% reads a txt file produced by to_lex1.pl
filename = 'wiktionary_bilexicon_en-es';
X = importdata([filename,'.txt']);

fid = fopen([filename,'.txt']);
i = 1;
while 1 % read lines one by one.
    line = fgetl(fid);
    if ~ischar(line) 
        break 
    end
    matches = split(line, ',');
    word1   = matches{1};
    word2   = matches{2};
    B{i,1}  = word1;
    B{i,2}  = word2; 
    i = i + 1;
end

N = size(B,1);
save([filename, '.mat'], 'B');
fprintf('Saved bilexicon "%s.mat" with N=%d entries.\n', filename, N);