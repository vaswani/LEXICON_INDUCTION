function [pt,v] = randwalkP(M,T,p0)
    D = size(M,1); % dimension
    if nargin < 3
        pt = randperm(D);
    else
        pt = p0;
    end
    v = zeros(T,1);
    v(1) = f(pt, M);
    MAX_R = 10;
    miss_count = 0;
    MAX_MISS = 1000;
    for t=2:T,
        [pt,r] = findGoodPerm(pt,M,v(t-1),MAX_R);
        v(t) = f(pt,M);
        if r < MAX_R
            miss_count = 0;
        else
            miss_count = miss_count+1;
        end
        if miss_count == MAX_MISS
            break
        end
        
    end
    v = v(1:t);
end

function v=f(q,M)
    Z=toPermMatrix(q);
    %assert(all(sum(Z)==1));
    %assert(all(sum(Z')==1));
    
    v = sum(sum(Z' .* M));
    %assert(v2-v<1e-10);
end

function Z=toPermMatrix(q)
    D = length(q);
    Z = sparse(zeros(D));
    Z(q + (0:D:D^2-1)) = 1;
end

function [bestq,r]=findGoodPerm(p,M,v,MAX_R)
    D = length(p);
    bestq = p;
    R = MAX_R;
    I=randperm(D);
    for r=1:floor(R),
        i = I(r);
        j = I(r+1);
        k = I(r+2);
        q = p;
        q(j) = p(i);
        q(i) = p(k);
        q(k) = p(j);
        if length(unique(q))~=D
            keyboard
        end
        if f(q,M) < v
            bestq = q;
            break; % found a minimizing one. return it.
        end
    end
end

