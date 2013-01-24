classdef Junk
    %JUNK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        %     methods(Static, Access=private)
%         
%         
%         
%         function model = assignModel(model, S, U, M, mu, i, d)
%             range = 1:d;
%             U = U(:,range);
%             M = M(range, range);
%             
%             model.W{i}   = S * U*M;
%             model.Psi{i} = S - model.W{i}*model.W{i}';
%             model.mu{i}  = mu;
%             model.M{i}   = M;
%             model.U{i}   = U;
%         end
%         
%         function latentX = computeExpectedLatent(model, X, i)
%             M = model.M{i};
%             U = model.U{i};
%             mu = model.mu{i};
%             latentX = M'*U'*X{i}';
%             latentX = bsxfun(@minus, latentX, M'*(U'*mu'))';
%             % see page 4 of paper.
%             % E[z|x_n] = M_i' U_i' (x_n - mu_i)
%         end   
%     end
    end
    
end

% computes the probabilistic PCA model, given X and Y.
%             [N1,D1] = size(X{1});
%             [N2,D2] = size(X{2});
%             assert(N1==N2);
%             % compute entire covariance matrix.
%             C_XX = C(1:D1, 1:D1);
%             C_YY = C(D1+(1:D2), D1+(1:D2));
%             C_XY = C(1:D1, D1+(1:D2));
%             mu_X = mu(1:D1);
%             mu_Y = mu(D1+(1:D2));
% 
%             Z_X = inv(chol(C_XX)); 
%             Z_Y = inv(chol(C_YY));
%             
%             % compute U's and M's
%             [V_X, P, V_Y] = svd(Z_X' * C_XY * Z_Y); % TODO: should it be Z_X or Z_X'?
%             M = diag(sqrt(diag(P))); % use the same M for the two "sides". This solution minimizes the conditional entropy of x given z.
%             U_X = Z_X * V_X;  
%             U_Y = Z_Y * V_Y; % should this be V_Y' ?
%           
%             %% debugging
%             if isfield(options, 'debug') && options.debug ~= 0
%                 temp1 = U_X'*C_XX*U_X; v1 = norm(temp1 - eye(D1));
%                 temp2 = U_Y'*C_YY*U_Y; v2 = norm(temp2 - eye(D2));
%                 temp_12 = U_X'*C_XY*U_Y; v3 = norm(temp_12 - P);
%                 if v1+v2+v3 > 1e-8
%                     fprintf('an assertion failed [%f,%f,%f].\n', v1, v2, v3);
%                     keyboard
%                 else
%                     fprintf('All assertions passed.\n');
%                 end
%                 
%             end
%             
%             % TODO: compare solution against online code.
%             d = options.d;
%             model = struct();
%             model = CCAUtil.assignModel(model, C_XX, U_X, M, mu_X, 1, d);
%             model = CCAUtil.assignModel(model, C_YY, U_Y, M, mu_Y, 2, d);
%             model.options = options;
%             model.P = diag(P); model.P = model.P(1:d);



%  function latent=expectedLatent(model, X)
%             % returns the expected latent representation Z.X, Z.Y given the model
%             % what is the dimensionality of Z ?
%             latent{1} = CCAUtil.computeExpectedLatent(model, X, 1);
%             latent{2} = CCAUtil.computeExpectedLatent(model, X, 2);
%         end

