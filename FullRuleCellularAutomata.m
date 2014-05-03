classdef FullRuleCellularAutomata < handle
% A FullRuleCellularAutomata object stores the full rule table for every
% possible neighborhood configuration.  Assumes 5-neighborhood, 2d CA with
% periodic boundaries and binary states.
    properties
        rule; % Rule table
        neighbors; % N x 5 neihbor indices (including self)
        K; % number of states
        a; % column vector of unit activations
    end
    methods
        function frca = FullRuleCellularAutomata(rule, neighbors, K)
        % OuterTotalisticCellularAutomata constructs a cellular automata
        % object with the given rule table and dimensions.
        
            if nargin > 0 % check for object array construction
                frca.rule = rule;
                frca.neighbors = neighbors;
                frca.K = K;
                frca.a = zeros(size(neighbors,1), 1);
            end
            
        end
        function pulse(frca, x, b)
            
            % Localize variables
            a = frca.a;
            N = numel(a);
            rule = frca.rule;
            neighbors = frca.neighbors;
            K = frca.K;
            pows = K.^(0:4)'; % conversion to base K
            
            % Get neighborhood states
            neighborhood = reshape(a(neighbors(:)),N,5);
            % Apply rules
            a = rule(neighborhood*pows+1);
            % Feedback (Map to node index) for next round
            a(ceil(N*(tanh(b)+1)/2)) = K-1;
            
            frca.a = a;
        end
        function [fit, readOut, Y] = fitness(frca, X, T)
        % Inline fitness evaluation of otcas on mackey-glass

            % Localize variables
            a = zeros(size(frca.a));
            N = numel(a);
            rule = frca.rule;
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
                % Feedback (Map to node index)
                a(ceil(N*(tanh(T(:,t))+1)/2)) = K-1;
                % Get neighborhood states
                neighborhood = reshape(a(neighbors(:)),N,5);
                % Apply rules
                a = rule(neighborhood*pows+1);
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
                % Feedback
                a(ceil(N*(tanh(y)+1)/2)) = 1;
                % Get neighborhood states
                neighborhood = reshape(a(neighbors(:)),N,5);
                % Apply rules
                a = rule(neighborhood*pows+1);
                % Output
                y = readOut*A(:,t);
            end
            
            % Mean squared testing error
            err = (T - Y).^2;
            err = mean(err(:));
            fit = 1/err;
        end
        function clone = copy(frca)
            clone = FullRuleCellularAutomata;
            clone.rule = frca.rule;
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
        function frca = random(dims,K)
        % random constructs a randomized FullRuleCellularAutomata.
        
            rule = randi(K, [K^5, 1])-1;
            neighbors = FullRuleCellularAutomata.makeNeighbors(dims);
            frca = FullRuleCellularAutomata(rule, neighbors, K);
            
        end
        % Gen ops
        function [child1, child2] = crossover(parent1, parent2)
        % Cross-over rule tables, assuming equal K
            cutpt = randi(parent1.K-1);
            rule1 = [parent1.rule(1:cutpt); parent2.rule(cutpt+1:end)];
            rule2 = [parent2.rule(1:cutpt); parent1.rule(cutpt+1:end)];
            
            % Wrap child rules in frca objects
            child1 = FullRuleCellularAutomata(rule1, parent1.neighbors, parent1.K);
            child2 = FullRuleCellularAutomata(rule2, parent2.neighbors, parent2.K);
            
        end
        function child = mutate(parent, mutation_rate)
            rule = parent.rule;
            mutations = rand(size(parent.rule)) < mutation_rate;
            rule(mutations) = randi(parent.K-1, nnz(mutations), 1);
            child = FullRuleCellularAutomata(rule, parent.neighbors, parent.K); % wrap
        end
    end
end