classdef CCAUtil
   
    methods(Static)
        % Reference: "A probabilistic Interpretation of Canonical
        % Correlation Analysis". 
        % Bach and Jordan 2006.
        % THM 2:
        % assuming generative model:
        % z ~ N(0,Id) for min(d_1,d_2) > d >= 1, 
        % x_i|z ~ N(W_iz + mu_i, \Psi_i), i=1,2
        %
        % where d_i is the dimension of samples in X_i
        
        function model=latentCCA(X,Y,options)
            max_d = min(size(X,2), size(Y,2));
            if options.d > max_d,
                fprintf('Warning: options.d > max_d (%d > %d) - truncating to %d.\n', options.d, max_d, max_d);
                options.d = max_d;
            end
            %% Computes the CCA model, given X and Y.
            [U_X,U_Y,p] = canoncorr(X, Y);
            % note that we do not compute nor use the covariance matrices
            % defined in the probabilistic model.
          
            %% Compute the latent representation of X, Y, in dimension d.
            range = 1:min(options.d,length(p));
            model.M = diag(sqrt(p(range)));
            model.U_X = U_X(:,range);
            model.U_Y = U_Y(:,range);
            model.p = p;
        end

        function Z=getLatent(model, X, Y)
            Z.X = CCAUtil.getSingleLatent(X, model.U_X, model.M);
            Z.Y = CCAUtil.getSingleLatent(Y, model.U_Y, model.M);
            %latent.X = CCAUtil.getLatent(X, U_X, M); -- old implementation
            %latent.Y = CCAUtil.getLatent(Y, U_Y, M); -- old implementation
            % Note that canoncorr() can return similar U,V as 4&5th outputs.
            % then, the following was verified to be negligible:
            % norm(inv(M')*latent.Y' - V'), norm(inv(M')*latent.X' - U')
            % however this implementation may be more efficient, when 
            % options.d < max_d
        end
        
        function Z=getSingleLatent(X,U,M)
            mu = mean(X);
            Z = M'*U'*X';
            Z = bsxfun(@minus, Z, M'*(U'*mu'))';
        end
        
         function [Wx, Wy, r] = Hardoon(X,Y,tau)
            % CCA calculate canonical correlations
            %
            % [Wx Wy r] = cca(X,Y) where Wx and Wy contains the canonical correlation
            % vectors as columns and r is a vector with corresponding canonical
            % correlations.
            %
            % Update 31/01/05 added bug handling.
            if nargin < 3 
                tau = 10^-8;
            end

            if (nargin < 2)
              disp('Inocorrect number of inputs');
              help cca;
              Wx = 0; Wy = 0; r = 0;
              return;
            end

            % calculating the covariance matrices
            z = [X; Y];
            C = cov(z.');
            sx = size(X,1);
            sy = size(Y,1);
            Cxx = C(1:sx, 1:sx) + tau*eye(sx);
            Cxy = C(1:sx, sx+1:sx+sy);
            Cyx = Cxy';
            Cyy = C(sx+1:sx+sy,sx+1:sx+sy) + tau*eye(sy);

            %calculating the Wx cca matrix
            Rx = chol(Cxx);
            invRx = inv(Rx);
            Z = invRx'*Cxy*(Cyy\Cyx)*invRx;
            Z = 0.5*(Z' + Z);  % making sure that Z is a symmetric matrix
            [Wx,r] = eig(Z);   % basis in h (X)
            r = sqrt(real(r)); % as the original r we get is lamda^2
            Wx = invRx * Wx;   % actual Wx values

            % calculating Wy
            Wy = (Cyy\Cyx) * Wx; 

            % by dividing it by lamda
            Wy = Wy./repmat(diag(r)',sy,1);
         end
         
         function [U,V,r] = Haghighi(X,Y, tau)
             if nargin < 3 
                tau = 10^-8;
             end
            
             [Nx, Dx] = size(X)
             [Ny, Dy] = size(Y)

            if (nargin < 2)
              disp('Inocorrect number of inputs');
              help cca;
              Wx = 0; Wy = 0; r = 0;
              return;
            end
            
            Cxx = zeros(Dx, Dx);
            Cyy = zeros(Dy, Dy);
            Cxy = zeros(Dx, Dy);
            Cxy = X'*Y / N;
            Cxx = X'*X / N;
            Cyy = Y'*Y / N;
            Cyx = Cxy';
            
            Bx = (1-tau) * Cxx + tau*eye(Dx);
            By = (1-tau) * Cyy + tau*eye(Dy);
            
            BC = By \ Cyx;
            E = (Bx \ Cxy) * BC;
            
            [U,S,V] = eig(E);
            lambda = sqrt(diag(S) + 1e-10)
         end
         
         function [Wx, Wy, r, E] = Magnus2(X,Y)
             % CCA calculate canonical correlations
            %
            % [Wx Wy r] = cca(X,Y) where Wx and Wy contains the canonical correlation
            % vectors as columns and r is a vector with corresponding canonical
            % correlations. The correlations are sorted in descending order. X and Y
            % are matrices where each column is a sample. Hence, X and Y must have
            % the same number of columns.
            %
            % Example: If X is M*K and Y is N*K there are L=MIN(M,N) solutions. Wx is
            % then M*L, Wy is N*L and r is L*1.
            %
            %
            % © 2000 Magnus Borga, Linköpings universitet

            % --- Calculate covariance matrices ---

            z = [X;Y];
            C = cov(z.');
            sx = size(X,1);
            sy = size(Y,1);
            Cxx = C(1:sx, 1:sx) + 10^(-8)*eye(sx);
            Cxy = C(1:sx, sx+1:sx+sy);
            Cyx = Cxy';
            Cyy = C(sx+1:sx+sy, sx+1:sx+sy) + 10^(-8)*eye(sy);
            invCyy = inv(Cyy);
            By = (Cyy\Cyx);
            % --- Calcualte Wx and r ---
            E = (Cxx\Cxy)*By;
            [Wx,r] = eig(E); % Basis in X
            r = sqrt(real(r));      % Canonical correlations

            % --- Sort correlations ---

            V = fliplr(Wx);		% reverse order of eigenvectors
            r = flipud(diag(r));	% extract eigenvalues anr reverse their orrer
            [r,I]= sort((real(r)));	% sort reversed eigenvalues in ascending order
            r = flipud(r);		% restore sorted eigenvalues into descending order
            for j = 1:length(I)
              Wx(:,j) = V(:,I(j));  % sort reversed eigenvectors in ascending order
            end
            Wx = fliplr(Wx);	% restore sorted eigenvectors into descending order

            % --- Calcualte Wy  ---

            Wy = invCyy*Cyx*Wx;     % Basis in Y
            Wy = Wy./repmat(sqrt(sum(abs(Wy).^2)),sy,1); % Normalize Wy
         end
         
         function [Wx, Wy, r, E] = Magnus(X,Y)
            % CCA calculate canonical correlations
            %
            % [Wx Wy r] = cca(X,Y) where Wx and Wy contains the canonical correlation
            % vectors as columns and r is a vector with corresponding canonical
            % correlations. The correlations are sorted in descending order. X and Y
            % are matrices where each column is a sample. Hence, X and Y must have
            % the same number of columns.
            %
            % Example: If X is M*K and Y is N*K there are L=MIN(M,N) solutions. Wx is
            % then M*L, Wy is N*L and r is L*1.
            %
            %
            % © 2000 Magnus Borga, Linköpings universitet

            % --- Calculate covariance matrices ---

            z = [X;Y];
            C = cov(z.');
            sx = size(X,1);
            sy = size(Y,1);
            Cxx = C(1:sx, 1:sx) + 10^(-8)*eye(sx);
            Cxy = C(1:sx, sx+1:sx+sy);
            Cyx = Cxy';
            Cyy = C(sx+1:sx+sy, sx+1:sx+sy) + 10^(-8)*eye(sy);
            invCyy = inv(Cyy);

            % --- Calcualte Wx and r ---
            E = inv(Cxx)*Cxy*invCyy*Cyx;
            [Wx,r] = eig(E); % Basis in X
            r = sqrt(real(r));      % Canonical correlations

            % --- Sort correlations ---

            V = fliplr(Wx);		% reverse order of eigenvectors
            r = flipud(diag(r));	% extract eigenvalues anr reverse their orrer
            [r,I]= sort((real(r)));	% sort reversed eigenvalues in ascending order
            r = flipud(r);		% restore sorted eigenvalues into descending order
            for j = 1:length(I)
              Wx(:,j) = V(:,I(j));  % sort reversed eigenvectors in ascending order
            end
            Wx = fliplr(Wx);	% restore sorted eigenvectors into descending order

            % --- Calcualte Wy  ---

            Wy = invCyy*Cyx*Wx;     % Basis in Y
            Wy = Wy./repmat(sqrt(sum(abs(Wy).^2)),sy,1); % Normalize Wy
         end
         
         function testMagnus(A,B)
             [Wx2, Wy2, r2, E2] = CCAUtil.Magnus2(A,B);
             [Wx, Wy, r, E] = CCAUtil.Magnus(A,B);
             
             norm_E = norm(E-E2, 'fro')
             I2 = (r2>1e-6);
             I  = (r>1e-6);
             Wx2 = Wx2(:, I2);
             Wx  = Wx (:, I);
             Wy2 = Wy2(:, I2);
             Wy  = Wy (:, I);
             
             er = norm(r-r2);
             ex = norm(Wx-Wx2);
             ey = norm(Wy-Wy2);

             [er, ex, ey]
         end
         
         function R=symSqrt(A)
             [U,S] = eig(A);
             R = U*sqrt(S)*U';
             R = (R+R')/2;
         end
         
         function pcca(X,Y)
            z = [X,Y];
            C = cov(z);
            tau = 0.01;
            sx = size(X,2);
            sy = size(Y,2);
            Cxx = (1-tau)*C(1:sx, 1:sx) + tau*eye(sx);
            Cxy = C(1:sx, sx+1:sx+sy);
            Cyx = Cxy';
            Cyy = (1-tau)*C(sx+1:sx+sy, sx+1:sx+sy) + tau*eye(sy);
            
            Rx = CCAUtil.symSqrt(Cxx);
            Ry = CCAUtil.symSqrt(Cyy);
            
            E = inv(Rx)*Cxy*inv(Ry);
            [Vx, S, Vy]=svd(E);
            
            Ux = Rx \ Vx;
            Uy = Ry \ Vy;
            p = diag(S)
            
            Jx = Ux' * Cxx * Ux;
            Jy = Uy' * Cyy * Uy;
         end
         
         function testCCA()
             D = 3;
             Z = diag([1,2,3]);
             X = Z * randn(3,3);
             Y = Z * randn(3,3);
             
             [Wx, Wy, r] = canoncorr(X,Y);
             
             Cxx = cov(X);
             Cyy = cov(Y);
             
             Exx = Wx'*Cxx * Wx;
             Eyy = Wy'*Cyy * Wy;
             
             real(Exx)
             real(Eyy)
             
         end
         
         

    end
end

