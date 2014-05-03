classdef HeterogeneousCellularAutomata < handle
% A HeterogenousCellularAutomata object is like FullRuleCellularAutomata,
% except a single CA can have different rule sets for different nodes.
    properties
        rules; % cell array of rule tables
        followers; % followers{i} = units that follow rules{i}
        neighbors; % N x 5 neighbor indices (including self)
        K; % number of states
        a; % column vector of unit activations
    end
    methods
        function frca = HeterogeneousCellularAutomata(rules, followers, neighbors, K)
        % HeterogeneousCellularAutomata constructs a cellular automata
        % object with the given rule table and dimensions.
        
            if nargin > 0 % check for object array construction
                frca.rules = rules;
                frca.followers = followers;
                frca.neighbors = neighbors;
                frca.K = K;
                frca.a = zeros(size(neighbors,1), 1);
            end
            
        end
        function pulse(frca, x, b)
            
            % Localize variables
            a = frca.a;
            N = numel(a);
            rules = frca.rules;
            followers = frca.followers;
            neighbors = frca.neighbors;
            K = frca.K;
            pows = K.^(0:4)'; % conversion to base K
            
            % Feedback (Map to node index) for next round
            a(ceil(N*(tanh(b)+1)/2)) = K-1;
            % Get neighborhood states
            neighborhood = reshape(a(neighbors(:)),N,5);
            % Apply rules
            for i = 1:numel(followers)
                a(followers{i}) = rules{i}(neighborhood(followers{i},:)*pows+1);
            end
            frca.a = a;
        end
        function [fit, readOut, Y] = fitness(frca, X, T)
        % Inline fitness evaluation of otcas on mackey-glass

            % Localize variables
            a = zeros(size(frca.a));
            N = numel(a);
            rules = frca.rules;
            followers = frca.followers;
            neighbors = frca.neighbors;
            K = frca.K;
            pows = K.^(0:4)'; % conversion to base K
            len = size(T,2);

            % Generate training data
            % Pre-allocate records
            A = zeros(N, len);
            % Stream inputs
            for t = 1:len
                % Record activations
                A(:,t) = a;
                % Feedback (Map to node index) for next round
                a(ceil(N*(tanh(b)+1)/2)) = K-1;
                % Get neighborhood states
                neighborhood = reshape(a(neighbors(:)),N,5);
                % Apply rules
                for i = 1:numel(followers)
                    a(followers{i}) = rules{i}(neighborhood(followers{i},:)*pows+1);
                end
            end

            % Ridge regression on readout
            tr = len/4:3*len/4;
            [U,S,V] = svd(A(:,tr),'econ');
            S = diag(S);
            ridge = 10;
            D = diag(S./(S.^2+ridge));
            readOut = T(:,tr+1)*V*D*U';

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
                a(ceil(N*(tanh(b)+1)/2)) = K-1;
                % Get neighborhood states
                neighborhood = reshape(a(neighbors(:)),N,5);
                % Apply rules
                for i = 1:numel(followers)
                    a(followers{i}) = rules{i}(neighborhood(followers{i},:)*pows+1);
                end
                % Output
                y = readOut*A(:,t);
            end
            
            % Mean squared testing error
            err = (T - Y).^2;
            err = mean(err(:));
            fit = 1/err;
        end
        function clone = copy(frca)
            clone = HeterogeneousCellularAutomata;
            clone.rules = frca.rules;
            clone.followers = frca.followers;
            clone.neighbors = frca.neighbors;
            clone.K = frca.K;
            clone.a = frca.a;
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
        
            rules = {randi(K, [K^5, 1])-1};
            neighbors = FullRuleCellularAutomata.makeNeighbors(dims);
            followers = {1:size(neighbors,1)};
            hca = FullRuleCellularAutomata(rule, followers, neighbors, K);
            
        end
        % Gen ops
        function [child1, child2] = crossover(parent1, parent2)
        % Cross-over units
            leaders1 = zeros(size(parent1.neighbors,1),1);
            for i = 1:numel(parent1.followers)
                leaders1(parent1.followers{i}) = i;
            end
            leaders2 = zeros(size(parent2.neighbors,1),1);
            for i = 1:numel(parent2.followers)
                leaders1(parent2.followers{i}) = i;
            end
            followers2 = zeros(size(parent2.neighbors,1),1);
            cutpt = randi(parent1.K-1);
            rule1 = [parent1.rule(1:cutpt); parent2.rule(cutpt+1:end)];
            rule2 = [parent2.rule(1:cutpt); parent1.rule(cutpt+1:end)];
            
            % Wrap child rules in frca objects
            child1 = FullRuleCellularAutomata(rule1, parent1.neighbors, parent1.K);
            child2 = FullRuleCellularAutomata(rule2, parent2.neighbors, parent2.K);
            
        end
        function child = mutate(parent, mutation_rate)
            child = HeterogeneousCellularAutomata;
            child.rules = parent.rules;
            for i = 1:numel(child.rules)
                mutations = rand(size(child.rules{i})) < mutation_rate;
                child.rules{i}(mutations) = randi(parent.K-1, nnz(mutations), 1);
            end
            child.followers = parent.followers;
            child.K = parent.K;
            child.neighbors = parent.neighbors;
        end
    end
end