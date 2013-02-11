classdef Util

    methods(Static)
        
        function inv_pi = inverse_perm(pi)
            inv_pi(pi) = 1:length(pi);
            % e.g pi = randperm(5); sig=Util.inverse_perm(pi); pi(sig)
            % should output [1 2 3 4 5]
        end
        
        function v=hamming(p,q)
            v=sum(p~=q);
        end
        
        function is_perm(pi)
            assert(all(1:length(pi)==(unique(pi)))); % assert pi is indeed a permutation.
        end
        
        % random swap k elements in pi
        function pi=randswap(pi, k)
            q = pi;
            i = randperm(length(q));
            i = i(1:ceil(k)); % take k distinct elements.
            if k==2
                pi(i) = pi(i(end:-1:1));
            else
                pi(i) = q(sort(i));
            end
        end
        
        function dim = mass_to_dim(X, singular_mass)
            % finds the number of dimensions in X
            % that whose mass in terms of singular values is at least p
            if singular_mass == 1
                dim = size(X,2);
            else
                q = svd(X); 
                dim = find(cumsum(q)/sum(q)>singular_mass,1);
            end
        end
        
        function d=edit_distance_levenshtein(s,t)
            % EDIT_DISTANEC_LEVENSHTEIN calculates the Levenshtein edit distance.
            %
            % This code is part of the work described in [1]. In [1], edit distances
            % are applied to match linguistic descriptions that occur when referring
            % to objects (in order to achieve joint attention in spoken human-robot /
            % human-human interaction).
            %
            % [1] B. Schauerte, G. A. Fink, "Focusing Computational Visual Attention
            %     in Multi-Modal Human-Robot Interaction," in Proc. ICMI,  2010.
            %
            % @author: B. Schauerte
            % @date:   2010
            % @url:    http://cvhci.anthropomatik.kit.edu/~bschauer/

            % Copyright 2010 B. Schauerte. All rights reserved.
            % 
            % Redistribution and use in source and binary forms, with or without 
            % modification, are permitted provided that the following conditions are 
            % met:
            % 
            %    1. Redistributions of source code must retain the above copyright 
            %       notice, this list of conditions and the following disclaimer.
            % 
            %    2. Redistributions in binary form must reproduce the above copyright 
            %       notice, this list of conditions and the following disclaimer in 
            %       the documentation and/or other materials provided with the 
            %       distribution.
            % 
            % THIS SOFTWARE IS PROVIDED BY B. SCHAUERTE ''AS IS'' AND ANY EXPRESS OR 
            % IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
            % WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
            % DISCLAIMED. IN NO EVENT SHALL B. SCHAUERTE OR CONTRIBUTORS BE LIABLE 
            % FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
            % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
            % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
            % BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
            % WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
            % OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
            % ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
            % 
            % The views and conclusions contained in the software and documentation
            % are those of the authors and should not be interpreted as representing 
            % official policies, either expressed or implied, of B. Schauerte.

            m=numel(s);
            n=numel(t);

            d=zeros(m+1,n+1);

            % initialize distance matrix
            for i=0:m % deletion
                d(i+1,1)=i;
            end
            for j=0:n % insertion
                d(1,j+1)=j;
            end

            for j=2:n+1
                for i=2:m+1
                    if s(i-1) == t(j-1)
                    d(i,j)=d(i-1,j-1);
                    else
                    d(i,j)=min([ ...
                      d(i-1,j) + 1, ...  % deletion
                      d(i,j-1) + 1, ...  % insertion
                      d(i-1,j-1) + 1 ... % substitution
                      ]);
                    end
                end
            end

            d=d(m+1,n+1);
        end
        
        function G = to_stochastic_graph(G)
            G = (G + G')/2; % make the graph symmetric.
            g = sum(G,2);
            g(g==0)=1;
            g = 1./g;       % divide by degree.
            G = sparse(bsxfun(@times, G, g));
        end
        
        function v=strlen(strings)
            if isstr(strings)
                v=length(strings);
            end
            if iscell(strings) 
                N = length(strings);
                v = zeros(N,1);
                i=1;
                for I = 1:N
                    v(i) = length(strings{i});
                    i = i  +1;
                end
            end
        end
    end
end

