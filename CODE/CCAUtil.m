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
        
        function [latent,p]=latentCCA(X,Y,options)
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
            M = diag(sqrt(p(range)));
            U_X = U_X(:,range);
            U_Y = U_Y(:,range);
            
            latent.X = CCAUtil.getLatent(X, U_X, M);
            latent.Y = CCAUtil.getLatent(Y, U_Y, M);
            % Note that canoncorr() can return similar U,V as 4&5th outputs.
            % then, the following was verified to be negligible:
            % norm(inv(M')*latent.Y' - V'), norm(inv(M')*latent.X' - U')
            % however this implementation may be more efficient, when 
            % options.d < max_d
        end
        
        function Z=getLatent(X,U,M)
            mu = mean(X);
            Z = M'*U'*X';
            Z = bsxfun(@minus, Z, M'*(U'*mu'))';
        end
    end
end

