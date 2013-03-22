classdef PCAUtil

    methods(Static)
        function model=PCA(X)
            N = size(X,1); 
            mu = mean(X);
            X = Util.sub(X, mu);    % subtract off the mean for each dimension 
            
            Sigma = (X'*X)/N;       % calculate the covariance matrix 
            [V, S] = eig(Sigma);    % find the eigenvectors and eigenvalues 
            
            S = diag(S);            % extract diagonal of matrix as vector 
            [~, I] = sort(-S);    % sort the variances in decreasing order 

            model = struct('mu', mu, 'S', diag(S(I)), 'V', V(:,I)); %S=e.vectors, V=e.values
        end
        
        function X = center(model, X)
            X = Util.sub(X, model.mu);
        end
        
        function Z=project(model, X, d)
            if nargin < 3
                d = size(model.V,2);
            end
            % compute X*Vd where Vd are the first d eigen vectors.
            % we obtain an Nxd matrix Z.
            X = PCAUtil.center(model, X);
            Z = X*model.V(:, 1:d);
        end
        
        function X=axis_align(model, X, d)
            if nargin < 3
                d = size(X,2);
            end
            
            s = diag(model.S);      % reduce the dimension
            s = s(1:d);
            X = PCAUtil.project(model, X, d);
        end
        
        function X=whiten(model, X, d)
            if nargin < 3
                d = size(X,2);
            end
            
            s = diag(model.S);      % reduce the dimension
            s = s(1:d);
            T = diag(1./sqrt(s));   
            X = PCAUtil.project(model, X, d)*T;
        end
        
        function dim = getEigenmassDim(model, eigenmass)
            s = diag(model.S);
            mass = cumsum(s)/sum(s);
            dim = find(mass>eigenmass,1);
        end
        
        function plot(model, X, labels, d)
            if ~iscell(X)
                X = {X};
                labels = {labels};
            end
            for i=1:length(X),
                Zi = PCAUtil.whiten(model, X{i}, d); % project using model
                PCAUtil.myscatter(Zi, labels{i}, d);
            end
        end
        
        function myscatter(Zi, labels, d, colors, S)
            if size(labels,1)~=1
                labels = labels';
            end
            Nc = length(labels);
            if nargin < 4
                colors = rand(Nc,3)*2/3+0.333;
                S = 50;
            end
            c = 0;
            for k = unique(labels),
                J = (labels == k);
                colorIndex = mod(c, Nc)+1;
                %color = colors{};
                color = colors(colorIndex, :);
                if d == 3,
                   % if mod(k,8)==0
                        scatter3(Zi(J, 1), Zi(J, 2), Zi(J, 3), S, color, 'filled');
                   % end
                else
                    scatter(Zi(J, 1), Zi(J, 2), S, color, 'filled');    
                end
                hold on;
                c = c + 1;
            end
        end
        
        function A=reducespec(A, p, mode)
            if p > 1
                error('p should in (0,1], p=%f',p);
            end
            [U,S,V] = svd(A);

            if nargin < 3 || strcmpi(mode, 'uniform')
                % reduces the eigenspectrum by p percent all across
                s = diag(S);
                q = sum(s)*(1-p) / length(s);
                s = max(s - q, 0);
                A = U*diag(s)*V';
            elseif strcmpi(mode,'bottom') % chop from the least (reduces rank quickly)
                r = sum(diag(S))*(1-p);
                j = min(size(A))+1;
                while r>0
                    j = j-1;
                    if r-S(j,j) > 0
                        r = r - S(j,j);
                        S(j,j) = 0;
                    else
                        S(j,j) = S(j,j)-r;
                        r = 0;
                    end
                end
                
                A = U*S*V';
            end            
        end
        
        function [Z, model] = zscore(X, model)
            if nargin < 2
                [Z,model.mu,model.Sigma] = zscore(X);
            else
                Z = bsxfun(@minus,X, model.mu);
                Sigma0 = model.Sigma;
                Sigma0(Sigma0==0) = 1;
                Z = bsxfun(@rdivide, Z, Sigma0);
            end
        end
        
        
        function A=evenspec(A, p)
            % transfer p percent of the eigenmass to all e.values
            [U,S,V] = svd(A);
            s = diag(S);
            e = ones(size(s));
            q = sum(s)/length(s);
            s = s*p + (1-p)*q*e;
            A = U*diag(s)*V';
        end
        
        function A = fixPSD(A, th)
            A = (A + A')/2; % ensure symmetric.
            [ispsd, evals] = PCAUtil.isPSD(A); 
            if ~ispsd
                if nargin < 2
                    th = 1e-4;
                end
                if min(evals) < th
                    A = A + th*eye(size(A));
                else
                    error('cannot fix A, magnitude of min e.val is too high.');
                end 
            end
        end
        
        function [b,d]=isPSD(A)
            [~,D] = eig(A);
            d = diag(D);
            b = all(d > 0);
        end
               
        function testplot()
            mu1 = [-1 2 1];                     % The mean vector.
            a = -40*pi/180;
            S1 = [cos(a) sin(a)/2, 0; sin(a)/2  cos(a)/2 0; 0 0 1]/2;
            mu2 = [0 -2 -2];                     % The mean vector.
            S2 = [0.5 0.2 0; 0.2 0.5 0; 0 0 0.8]/2;  % The covariance matrix.

            N = 100;               % The number of samples you want.

            % Generate the draws.
            data1 = mvnrnd(mu2, S1, N);
            data2 = mvnrnd(mu1, S1, N);
            data = [data1;data2];
            labels = [ones(N,1); 2*ones(N,1)]';
            model = PCAUtil.PCA(data);
            d = 3;
            figure;
            subplot(1,3,1);
            PCAUtil.myscatter(data, labels, d); title('data');
            subplot(1,3,2);
            PCAUtil.plot(model, {data}, {labels}, d); title('my model');
            [signals,~,~] = pca2(data');
            subplot(1,3,3);
            PCAUtil.myscatter(signals', labels, d); title('their model');
            figure; 
            biplot(PCAUtil.project(model, data, d));
        end
    end
end