classdef HeterogeneousCellularAutomata < handle
% A HeterogenousCellularAutomata object is like FullRuleCellularAutomata,
% except a single CA can have different rule sets for different nodes.
    properties
        rules; % rules(i,j) is new state for unit j with neighborhood configuration i
        neighbors; % N x 5 neighbor indices (including self)
        K; % number of states
        a; % column vector of unit activations
    end
    methods
        function hca = HeterogeneousCellularAutomata(rules, neighbors, K)
        % HeterogeneousCellularAutomata constructs a cellular automata
        % object with the given rule table and dimensions.
        
            if nargin > 0 % check for object array construction
                hca.rules = rules;
                hca.neighbors = neighbors;
                hca.K = K;
                hca.a = zeros(size(neighbors,1), 1);
            end
            
        end
        function pulse(hca, x, b)
            
            % Localize variables
            a = hca.a;
            N = numel(a);
            rules = hca.rules;
            neighbors = hca.neighbors;
            K = hca.K;
            pows = K.^(0:4)'; % conversion to base K
            
            % Feedback (Map to node index)
            a(ceil(N*(tanh(b)+1)/2)) = K-1;
            % Get neighborhood states
            neighborhoods = reshape(a(neighbors(:)),N,5);
            % Apply rules
            idx = neighborhoods*pows + (K^5)*(0:numel(a)-1)' + 1; 
            a = rules(idx);
            hca.a = a;
        end
        function [fit, readOut, Y] = fitness(hca, X, T)
        % Inline fitness evaluation of otcas on mackey-glass

            % Localize variables
            a = zeros(size(hca.a));
            N = numel(a);
            rules = hca.rules;
            neighbors = hca.neighbors;
            K = hca.K;
            pows = K.^(0:4)'; % conversion to base K
            len = size(T,2);

            % Generate training data
            % Pre-allocate records
            A = zeros(N, len);
            % Stream inputs
            for t = 1:len
                % Record activations
                A(:,t) = a;
                % Feedback (Map to node index)
                a(ceil(N*(tanh(T(:,t))+1)/2)) = K-1;
                % Get neighborhood states
                neighborhoods = reshape(a(neighbors(:)),N,5);
                % Apply rules
                idx = neighborhoods*pows + (K^5)*(0:numel(a)-1)' + 1; 
                a = rules(idx); % A(:,t+1) = fun( A(:,t), T(:,t) )
            end

            % Ridge regression on readout
            tr = len/4:3*len/4;
            [U,S,V] = svd(A(:,tr),'econ');
            S = diag(S);
            ridge = 10;
            D = diag(S./(S.^2+ridge));
            readOut = T(:,tr+1)*V*D*U'; % T(:,t+1) = fun( A(:,t) )

            % Re-stream using read-out
            Y = zeros(size(T));
            a(:) = 0;
            y = 0;
            for t = 1:len
                % Record
                A(:,t) = a;
                Y(:,t) = y;
                % Force
                if t < len/2, y = T(:,t); end;
                % Feedback (Map to node index) for next round
                a(ceil(N*(tanh(y)+1)/2)) = K-1;
                % Get neighborhood states
                neighborhoods = reshape(a(neighbors(:)),N,5);
                % Apply rules
                idx = neighborhoods*pows + (K^5)*(0:numel(a)-1)' + 1; 
                a = rules(idx); % A(:,t+1) = fun( A(:,t), T(:,t) )
                % Output
                y = readOut*A(:,t); % Y(:,t+1) = fun( A(:,t) )
            end
            
            % Mean squared testing error
            err = (T - Y).^2;
            err = mean(err(:));
            fit = 1/err;
        end
        function clone = copy(hca)
            clone = HeterogeneousCellularAutomata;
            clone.rules = hca.rules;
            clone.neighbors = hca.neighbors;
            clone.K = hca.K;
            clone.a = hca.a;
        end
    end
    methods(Static = true)
        function neighbors = makeNeighbors(dims)
        % makeNeighbors makes the neighbor matrix for given dimensions.
        % Parameters:
        %   dims : [rows, cols] is the grid size
        % Returns:
        %   neighbhors: the neighbor matrix
        
            % List (r,c) coordinates for each cell (0-based idx)
            R = dims(1); C = dims(2);
            r = repmat((0:R-1)',C,1);
            c = reshape(repmat(0:C-1, R, 1),[],1);
            % convert neighborhood to linear index
            neighbors = [...
                R*c + r,... %self
                R*mod(c-1,C) + r,... % left
                R*mod(c+1,C) + r,... % right
                R*c + mod(r-1,R),... % up
                R*c + mod(r+1,R) ...  % down
            ];
            % 1 based indexing
            neighbors = neighbors + 1;
            
        end
        function hca = random(dims, K)
        % random constructs a randomized HeterogeneousCellularAutomata.
        
            rules = randi(K, [K^5, prod(dims)])-1;
            neighbors = HeterogeneousCellularAutomata.makeNeighbors(dims);
            hca = HeterogeneousCellularAutomata(rules, neighbors, K);
            
        end
        % Gen ops
        function [child1, child2] = crossover(parent1, parent2)
            % Cross-over units
            cutpt = randi(numel(parent1.a)-1);
            rules1 = [parent1.rules(:,1:cutpt), parent2.rules(:,cutpt+1:end)];
            rules2 = [parent2.rules(:,1:cutpt), parent1.rules(:,cutpt+1:end)];
            
            % Wrap child rules in frca objects
            child1 = HeterogeneousCellularAutomata(rules1, parent1.neighbors, parent1.K);
            child2 = HeterogeneousCellularAutomata(rules2, parent2.neighbors, parent2.K);
            
        end
        function child = mutate(parent, mutation_rate)
            child = parent.copy();
            mutations = rand(size(parent.rules)) < mutation_rate;
            child.rules(mutations) = randi(parent.K-1, nnz(mutations), 1);
        end
    end
end