classdef MatchingUtil
    % MatchingUtil - a class for finding the minimum weighted matching
    % between a set of points.
    %
    % makeWeights - makes a matrix of weights between the data points in
    % data, and according to the options. 
    % 
    % match - computes the minimum matching given weights W.
    
    methods(Static)        
        function W=makeWeights(options, X, Y, GX, GY)
            N = size(GX,1);
            
%             UX = (eye(N) - options.lambda*GX) \ X;
%             UY = (eye(N) - options.lambda*GY) \ Y;
%             if strcmpi(options.weight_type, 'inner')
%                 W = UX*UY';
%                 W = max(W(:)) - W;
%             else
%                 W = pdist2(UX,UY,'euclidean');
%             end
             if strcmpi(options.weight_type, 'inner')
                U = X * Y';
                W = U;
                % use graph information
                for k=1:options.M,
                    W = W + options.lambda^k*(GX^k)*U*(GY^k)';
                end
                W = max(W(:)) - W;
             elseif strcmpi(options.weight_type, 'dist')
                 U = pdist2(X,Y,'euclidean');
                 W = U;
                 for k=1:options.M,
                     W = W + options.lambda^k*pdist2(GX^k*X,GY^k*Y);
                 end
             end
             

%             if isempty(options.weight_type) || strcmpi(options.weight_type, 'inner')
%                 % weights are proportional to the inner product of elements
%                 U = X * Y';
%                 W = U;
%                 for k=1:options.M,
%                     W = W + options.lambda^k*(GX^k)*U*(GY^k)';
%                 end
%                 W = max(W(:)) - W;
%             elseif strcmpi(options.weight_type, 'dist')
%                 % no need to subtract since we will look for the minimum matching
%                 W = pdist2(X,Y,'euclidean'); 
%             end
        end
        
        function match = init_matching(listA, listB, max_dist, max_seed)
            % find a good initial matching, based on edit distance.
            NA = length(listA);
            NB = length(listB);
            % calculate pairwise edit-distance
            W = Common.load_editdist_file(listA, listB);
            min_length = 0;
            max_length_diff = inf;
            if isempty(W)
                W = inf(NA,NB);
                for a=1:NA,
                    for b = 1:NB,
                        word_a = listA{a};
                        word_b = listB{b};
                        cond_min_length = length(word_a) > min_length && length(word_b) > min_length;
                        cond_length_diff = abs(length(word_a) - length(word_b)) < max_length_diff;
                        if cond_min_length && cond_length_diff % otherwise, skip.
                            W(a,b) = Util.edit_distance_levenshtein(word_a,word_b);
                        end
                    end
                    if mod(a, 100) == 0
                        fprintf('Done with %d / %d\n', a, NA);
                    end
                end
                
                Common.save_editdist_file(listA, listB, W);
            end
            % based on the distances, find a good matching.
            [best_match, total_cost, ec] = MatchingUtil.match(W);
            [sorted_ec, sigma] = sort(ec);
            best_match = best_match(sigma);
            top = find(sorted_ec>max_dist, 1)-1;
            top = min(max_seed, top);
            % store the indices of the top matches in source and target
            match.all.source = sigma;
            match.all.target = best_match;
            match.all.weights = sorted_ec;
            match.all.pi = [1:NA]';
            
            match.source = sigma(1:top);
            match.target = best_match(1:top);
            
            % [listA(match.source), listB(match.target), (mat2cell(sorted_ec(1:top)', ones(top,1)))]
            % [listA(match.all.source), listB(match.all.target), (mat2cell(sorted_ec', ones(NA,1)))]
        end
        
        function v = edge_cost(pi, cost)
            % computes the cost of a given edge set pi.
            N = length(pi);
            I = sub2ind(size(cost), 1:N, pi);
            v = cost(I);
        end
        
        function [pi, cost, edge_cost] = match(W, options)
            % calls lapjv to compute the **minimum** weighted matching
            if nargin < 2
                resolution = 1e-4;
            else
                resolution = max(2*0.5^options.t,1e-6);
            end
            fprintf('Matching with resolution=%f\n', resolution);
            [pi, cost] = MatchingUtil.lapjv(W, resolution);
            edge_cost = MatchingUtil.edge_cost(pi, W);
            if ~isinf(cost) && ~(sum(edge_cost)-cost < 1e-3);
                sum(edge_cost)-cost
                keyboard;
            end
        end
        
        function [rowsol,cost,v,u,costMat] = lapjv(costMat,resolution)
            % LAPJV  Jonker-Volgenant Algorithm for Linear Assignment Problem.
            %
            % [ROWSOL,COST,v,u,rMat] = LAPJV(COSTMAT) returns the optimal column indices,
            % ROWSOL, assigned to row in solution, and the minimum COST based on the
            % assignment problem represented by the COSTMAT, where the (i,j)th element
            % represents the cost to assign the jth job to the ith worker.
            % Other output arguments are:
            % v: dual variables, column reduction numbers.
            % u: dual variables, row reduction numbers.
            % rMat: the reduced cost matrix.
            %
            % For a rectangular (nonsquare) costMat, rowsol is the index vector of the
            % larger dimension assigned to the smaller dimension.
            %
            % [ROWSOL,COST,v,u,rMat] = LAPJV(COSTMAT,resolution) accepts the second
            % input argument as the minimum resolution to differentiate costs between
            % assignments. The default is eps.
            %
            % Known problems: The original algorithm was developed for integer costs.
            % When it is used for real (floating point) costs, sometime the algorithm
            % will take an extreamly long time. In this case, using a reasonable large
            % resolution as the second arguments can significantly increase the
            % solution speed.
            %
            % See also munkres, Hungarian

            % version 1.0 by Yi Cao at Cranfield University on 3rd March 2010
            % version 1.1 by Yi Cao at Cranfield University on 19th July 2010
            % version 1.2 by Yi Cao at Cranfield University on 22nd July 2010
            % version 2.0 by Yi Cao at Cranfield University on 28th July 2010
            % version 2.1 by Yi Cao at Cranfield University on 13th August 2010
            % version 2.2 by Yi Cao at Cranfield University on 17th August 2010

            % This Matlab version is developed based on the orginal C++ version coded
            % by Roy Jonker @ MagicLogic Optimization Inc on 4 September 1996.
            % Reference:
            % R. Jonker and A. Volgenant, "A shortest augmenting path algorithm for
            % dense and spare linear assignment problems", Computing, Vol. 38, pp.
            % 325-340, 1987.

            %
            % Examples
            % Example 1: a 5 x 5 example
            %{
            [rowsol,cost] = lapjv(magic(5));
            disp(rowsol); % 3 2 1 5 4
            disp(cost);   %15
            %}
            % Example 2: 1000 x 1000 random data
            %{
            n=1000;
            A=randn(n)./rand(n);
            tic
            [a,b]=lapjv(A);
            toc                 % about 0.5 seconds 
            %}
            % Example 3: nonsquare test
            %{
            n=100;
            A=1./randn(n);
            tic
            [a,b]=lapjv(A);
            toc % about 0.2 sec
            A1=[A zeros(n,1)+max(max(A))];
            tic
            [a1,b1]=lapjv(A1);
            toc % about 0.01 sec. The nonsquare one can be done faster!
            %check results
            disp(norm(a-a1))
            disp(b-b)
            %}

            if nargin<2
                maxcost=min(1e16,max(max(costMat)));
                resolution=eps(maxcost);
            end
            % Prepare working data
            [rdim,cdim] = size(costMat);
            M=min(min(costMat));
            if rdim>cdim
                costMat = costMat';
                [rdim,cdim] = size(costMat);
                swapf=true;
            else
                swapf=false;
            end
            dim=cdim;
            costMat = [costMat;2*M+zeros(cdim-rdim,cdim)];
            costMat(costMat~=costMat)=Inf;
            maxcost=max(costMat(costMat<Inf))*dim+1;
            if isempty(maxcost)
                maxcost = Inf;
            end
            costMat(costMat==Inf)=maxcost;
            % free = zeros(dim,1);      % list of unssigned rows
            % colist = 1:dim;         % list of columns to be scaed in various ways
            % d = zeros(1,dim);       % 'cost-distance' in augmenting path calculation.
            % pred = zeros(dim,1);    % row-predecessor of column in augumenting/alternating path.
            v = zeros(1,dim);         % dual variables, column reduction numbers.
            rowsol = zeros(1,dim)-1;  % column assigned to row in solution
            colsol = zeros(dim,1)-1;  % row assigned to column in solution

            if std(costMat(:)) < mean(costMat(:))
                numfree=0;
                free = zeros(dim,1);      % list of unssigned rows
                matches = zeros(dim,1);   % counts how many times a row could be assigned.
                % The Initilization Phase
                % column reduction
                for j=dim:-1:1 % reverse order gives better results
                    % find minimum cost over rows
                    [v(j), imin] = min(costMat(:,j));
                    if ~matches(imin)
                        % init assignement if minimum row assigned for first time
                        rowsol(imin)=j;
                        colsol(j)=imin;
                    elseif v(j)<v(rowsol(imin))
                        j1=rowsol(imin);
                        rowsol(imin)=j;
                        colsol(j)=imin;
                        colsol(j1)=-1;
                    else
                        colsol(j)=-1; % row already assigned, column not assigned.
                    end
                    matches(imin)=matches(imin)+1;
                end

                % Reduction transfer from unassigned to assigned rows
                for i=1:dim
                    if ~matches(i)      % fill list of unaasigned 'free' rows.
                        numfree=numfree+1;
                        free(numfree)=i;
                    else
                        if matches(i) == 1 % transfer reduction from rows that are assigned once.
                            j1 = rowsol(i);
                            x = costMat(i,:)-v;
                            x(j1) = maxcost;
                            v(j1) = v(j1) - min(x);
                        end
                    end
                end
            else
                numfree=dim-1;
                [v1 r]=min(costMat);
                free=1:dim;
                [~,c]=min(v1);
                imin=r(c);
                j=c;
                rowsol(imin)=j;
                colsol(j)=imin;
                % matches(imin)=1;
                free(imin)=[];
                x = costMat(imin,:)-v;
                x(j) = maxcost;
                v(j) = v(j) - min(x);
            end
            % Augmenting reduction of unassigned rows
            loopcnt = 0;
            while loopcnt < 2
                loopcnt = loopcnt + 1;
                % scan all free rows
                % in some cases, a free row may be replaced with another one to be scaed next
                k = 0;
                prvnumfree = numfree;
                numfree = 0;    % start list of rows still free after augmenting row reduction.
                while k < prvnumfree
                    k = k+1;
                    i = free(k);
                    % find minimum and second minimum reduced cost over columns
                    x = costMat(i,:) - v;
                    [umin, j1] = min(x);
                    x(j1) = maxcost;
                    [usubmin, j2] = min(x);
                    i0 = colsol(j1);
                    if usubmin - umin > resolution 
                        % change the reduction of the minmum column to increase the
                        % minimum reduced cost in the row to the subminimum.
                        v(j1) = v(j1) - (usubmin - umin);
                    else % minimum and subminimum equal.
                        if i0 > 0 % minimum column j1 is assigned.
                            % swap columns j1 and j2, as j2 may be unassigned.
                            j1 = j2;
                            i0 = colsol(j2);
                        end
                    end
                    % reassign i to j1, possibly de-assigning an i0.
                    rowsol(i) = j1;
                    colsol(j1) = i;
                    if i0 > 0 % ,inimum column j1 assigned easier
                        if usubmin - umin > resolution
                            % put in current k, and go back to that k.
                            % continue augmenting path i - j1 with i0.
                            free(k)=i0;
                            k=k-1;
                        else
                            % no further augmenting reduction possible
                            % store i0 in list of free rows for next phase.
                            numfree = numfree + 1;
                            free(numfree) = i0;
                        end
                    end
                end
            end

            % Augmentation Phase
            % augment solution for each free rows
            for f=1:numfree
                freerow = free(f); % start row of augmenting path
                % Dijkstra shortest path algorithm.
                % runs until unassigned column added to shortest path tree.
                d = costMat(freerow,:) - v;
                pred = freerow(1,ones(1,dim));
                collist = 1:dim;
                low = 1; % columns in 1...low-1 are ready, now none.
                up = 1; % columns in low...up-1 are to be scaed for current minimum, now none.
                % columns in up+1...dim are to be considered later to find new minimum,
                % at this stage the list simply contains all columns.
                unassignedfound = false;
                while ~unassignedfound
                    if up == low    % no more columns to be scaned for current minimum.
                        last = low-1;
                        % scan columns for up...dim to find all indices for which new minimum occurs. 
                        % store these indices between low+1...up (increasing up).
                        minh = d(collist(up));
                        up = up + 1;
                        for k=up:dim
                            j = collist(k);
                            h = d(j);
                            if h<=minh
                                if h<minh
                                    up = low;
                                    minh = h;
                                end
                                % new index with same minimum, put on index up, and extend list.
                                collist(k) = collist(up);
                                collist(up) = j;
                                up = up +1;
                            end
                        end
                        % check if any of the minimum columns happens to be unassigned.
                        % if so, we have an augmenting path right away.
                        for k=low:up-1
                            if colsol(collist(k)) < 0
                                endofpath = collist(k); 
                                unassignedfound = true;
                                break
                            end
                        end
                    end
                    if ~unassignedfound
                        % update 'distances' between freerow and all unscanned columns,
                        % via next scanned column.
                        j1 = collist(low);
                        low=low+1;
                        i = colsol(j1); %line 215
                        x = costMat(i,:)-v;
                        h = x(j1) - minh;
                        xh = x-h;
                        k=up:dim;
                        j=collist(k);
                        vf0 = xh<d;
                        vf = vf0(j);
                        vj = j(vf);
                        vk = k(vf);
                        pred(vj)=i;
                        v2 = xh(vj);
                        d(vj)=v2;
                        vf = v2 == minh; % new column found at same minimum value
                        j2 = vj(vf);
                        k2 = vk(vf);
                        cf = colsol(j2)<0; 
                        if any(cf) % unassigned, shortest augmenting path is complete.
                            i2 = find(cf,1);                
                            endofpath = j2(i2);
                            unassignedfound = true;
                        else 
                            i2 = numel(cf)+1;
                        end
                        % add to list to be scaned right away
                        for k=1:i2-1
                            collist(k2(k)) = collist(up);
                            collist(up) = j2(k);
                            up = up + 1;
                        end
                    end
                end
                % update column prices
                j1=collist(1:last+1);
                v(j1) = v(j1) + d(j1) - minh;
                % reset row and column assignments along the alternating path
                while 1
                    i=pred(endofpath);
                    colsol(endofpath)=i;
                    j1=endofpath;
                    endofpath=rowsol(i);
                    rowsol(i)=j1;
                    if (i==freerow)
                        break
                    end
                end
            end
            rowsol = rowsol(1:rdim);
            u=diag(costMat(:,rowsol))-v(rowsol)';
            u=u(1:rdim);
            v=v(1:cdim);
            cost = sum(u)+sum(v(rowsol));
            costMat=costMat(1:rdim,1:cdim);
            costMat = costMat - u(:,ones(1,cdim)) - v(ones(rdim,1),:);
            if swapf
                costMat = costMat';
                t=u';
                u=v';
                v=t;
            end
            if cost>maxcost
                cost=Inf;
            end
        end
        
        
        function [assignment,cost] = munkres(costMat)
            % MUNKRES   Munkres Assign Algorithm 
            %
            % [ASSIGN,COST] = munkres(COSTMAT) returns the optimal assignment in ASSIGN
            % with the minimum COST based on the assignment problem represented by the
            % COSTMAT, where the (i,j)th element represents the cost to assign the jth
            % job to the ith worker.
            %

            % This is vectorized implementation of the algorithm. It is the fastest
            % among all Matlab implementations of the algorithm.

            % Examples
            % Example 1: a 5 x 5 example
            %{
            [assignment,cost] = munkres(magic(5));
            [assignedrows,dum]=find(assignment);
            disp(assignedrows'); % 3 2 1 5 4
            disp(cost); %15
            %}
            % Example 2: 400 x 400 random data
            %{
            n=400;
            A=rand(n);
            tic
            [a,b]=munkres(A);
            toc                 % about 6 seconds 
            %}

            % Reference:
            % "Munkres' Assignment Algorithm, Modified for Rectangular Matrices", 
            % http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html

            % version 1.0 by Yi Cao at Cranfield University on 17th June 2008

            assignment = false(size(costMat));
            cost = 0;

            costMat(costMat~=costMat)=Inf;
            validMat = costMat<Inf;
            validCol = any(validMat);
            validRow = any(validMat,2);

            nRows = sum(validRow);
            nCols = sum(validCol);
            n = max(nRows,nCols);
            if ~n
                return
            end

            dMat = zeros(n);
            dMat(1:nRows,1:nCols) = costMat(validRow,validCol);

            %*************************************************
            % Munkres' Assignment Algorithm starts here
            %*************************************************

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   STEP 1: Subtract the row minimum from each row.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             dMat = bsxfun(@minus, dMat, min(dMat,[],2));

            %**************************************************************************  
            %   STEP 2: Find a zero of dMat. If there are no starred zeros in its
            %           column or row start the zero. Repeat for each zero
            %**************************************************************************
            zP = ~dMat;
            starZ = false(n);
            while any(zP(:))
                [r,c]=find(zP,1);
                starZ(r,c)=true;
                zP(r,:)=false;
                zP(:,c)=false;
            end

            while 1
            %**************************************************************************
            %   STEP 3: Cover each column with a starred zero. If all the columns are
            %           covered then the matching is maximum
            %**************************************************************************
                primeZ = false(n);
                coverColumn = any(starZ);
                if ~any(~coverColumn)
                    break
                end
                coverRow = false(n,1);
                while 1
                    %**************************************************************************
                    %   STEP 4: Find a noncovered zero and prime it.  If there is no starred
                    %           zero in the row containing this primed zero, Go to Step 5.  
                    %           Otherwise, cover this row and uncover the column containing 
                    %           the starred zero. Continue in this manner until there are no 
                    %           uncovered zeros left. Save the smallest uncovered value and 
                    %           Go to Step 6.
                    %**************************************************************************
                    zP(:) = false;
                    zP(~coverRow,~coverColumn) = ~dMat(~coverRow,~coverColumn);
                    Step = 6;
                    while any(any(zP(~coverRow,~coverColumn)))
                        [uZr,uZc] = find(zP,1);
                        primeZ(uZr,uZc) = true;
                        stz = starZ(uZr,:);
                        if ~any(stz)
                            Step = 5;
                            break;
                        end
                        coverRow(uZr) = true;
                        coverColumn(stz) = false;
                        zP(uZr,:) = false;
                        zP(~coverRow,stz) = ~dMat(~coverRow,stz);
                    end
                    if Step == 6
                        % *************************************************************************
                        % STEP 6: Add the minimum uncovered value to every element of each covered
                        %         row, and subtract it from every element of each uncovered column.
                        %         Return to Step 4 without altering any stars, primes, or covered lines.
                        %**************************************************************************
                        M=dMat(~coverRow,~coverColumn);
                        minval=min(min(M));
                        if minval==inf
                            return
                        end
                        dMat(coverRow,coverColumn)=dMat(coverRow,coverColumn)+minval;
                        dMat(~coverRow,~coverColumn)=M-minval;
                    else
                        break
                    end
                end
                %**************************************************************************
                % STEP 5:
                %  Construct a series of alternating primed and starred zeros as
                %  follows:
                %  Let Z0 represent the uncovered primed zero found in Step 4.
                %  Let Z1 denote the starred zero in the column of Z0 (if any).
                %  Let Z2 denote the primed zero in the row of Z1 (there will always
                %  be one).  Continue until the series terminates at a primed zero
                %  that has no starred zero in its column.  Unstar each starred
                %  zero of the series, star each primed zero of the series, erase
                %  all primes and uncover every line in the matrix.  Return to Step 3.
                %**************************************************************************
                rowZ1 = starZ(:,uZc);
                starZ(uZr,uZc)=true;
                while any(rowZ1)
                    starZ(rowZ1,uZc)=false;
                    uZc = primeZ(rowZ1,:);
                    uZr = rowZ1;
                    rowZ1 = starZ(:,uZc);
                    starZ(uZr,uZc)=true;
                end
            end

            % Cost of assignment
            assignment(validRow,validCol) = starZ(1:nRows,1:nCols);
            cost = sum(costMat(assignment));
        end

    end
    
end

