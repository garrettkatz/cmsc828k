classdef FullRuleCellularAutomata < handle
% A FullRuleCellularAutomata object stores the full rule table for every
% possible neighborhood configuration.  Assumes 5-neighborhood, 2d CA with
% periodic boundaries and binary states.
    properties
        rule; % Rule table
        neighbors; % N x 5 neihbor indices (including self)
        a; % column vector of unit activations
    end
    methods
        function frca = FullRuleCellularAutomata(rule, dims)
        % OuterTotalisticCellularAutomata constructs a cellular automata
        % object with the given rule table and dimensions.
        
            if nargin > 0 % check for object array construction
                frca.rule = rule;
                frca.neighbors = FullRuleCellularAutomata.makeNeighbors(dims);
                frca.a = zeros(size(grid,1), 1);
            end
            
        end
        function pulse(otca, x, b)
        % Pulse updates activations based on input vector x and feedback
        % vector b.

            K = otca.K;
            a = otca.a;

            % rescale external signals from R to [0,K]
            [xi,~,xs] = find(x);
            xs = K*(tanh(xs)+1)/2;
            [bi,~,bs] = find(b);
            bs = K*(tanh(bs)+1)/2;
            
            % Get neighborhood sum including external signals
            nsum = otca.grid*a;
            nsum(xi) = nsum(xi)+xs;
            nsum(bi) = nsum(bi)+bs;
            
            % Force to indices
            nsum = min(max(round(nsum), 0), 6*K);

            % Get linear indices in rule table
            idx = a + (K+1)*nsum + 1;

            % Apply rule
            otca.a = otca.rule(idx);
           
%             % MEX attempt (buggy)
%             nsum = otca.grid*otca.a;
%             new_a = otcapulse(nsum, otca.rule, otca.K, x, b, otca.a);
%             otca.a = new_a;
            
        end
        function fit = fitness(otca, X, T, tr)
        % Inline fitness evaluation of otcas on mackey-glass

            % Localize variables
            a = zeros(size(otca.a));
            N = numel(a);
            grid = otca.grid;
            rule = otca.rule;
            K = otca.K;

            % Connections (persistent to avoid confounding factors)
            persistent readIn;
            persistent readBack;
            if isempty(readIn)
                ext = randperm(N, 20); % indices of external-signal-receiving units
                readIn = sparse(ext(1:10), 1, 1, N, 1); % 1st 10 for input
                readBack = sparse(ext(11:20), 1, 1, N, 1); % last 10 for feedback
            end
            
            % Generate training data
            % Pre-allocate records
            A = zeros(N, size(X,2));
            % Stream inputs
            for t = 1:size(X,2)
                % Record activations
                A(:,t) = a;
                % Input and feedback
                x = readIn*X(:,t);
                b = readBack*T(:,t);
                % rescale external signals from reals to [0,K]
                [xi,~,xs] = find(x);
                xs = K*(tanh(xs)+1)/2;
                [bi,~,bs] = find(b);
                bs = K*(tanh(bs)+1)/2;
                % Get neighborhood sum including external signals
                nsum = grid*a;
                nsum(xi) = nsum(xi)+xs;
                nsum(bi) = nsum(bi)+bs;
                % Force to indices
                nsum = min(max(round(nsum), 0), 6*K);
                % Get linear indices in rule table
                idx = a + (K+1)*nsum + 1;
                % Apply rule
                a = rule(idx);
            end

            % Ridge regression on readout
            A = A(:,tr);
            T = T(:,tr+1);
            [U,S,V] = svd(A,'econ');
            S = diag(S);
            ridge = 10;
            D = diag(S./(S.^2+ridge));
            readOut = T*V*D*U';
            
            % Mean squared training error
            err = (T - readOut*A).^2;
            err = mean(err(:));
            fit = 1/err;
        end
        function clone = copy(otca)
            clone = OuterTotalisticCellularAutomata(otca.rule, otca.grid, otca.K);
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
            r = repmat((0:dims(1)-1)',dims(2),1);
            c = reshape(repmat(0:dims(2)-1, dims(1), 1),[],1);
        
        end
        function otca = random(dims, K)
        % random constructs a randomized OuterTotalisticCellularAutomata.
        
            rule = randi(K+1, [K+1, 6*K+1])-1; % 6 to include input/feedback
            grid = OuterTotalisticCellularAutomata.makeGrid(dims);
            otca = OuterTotalisticCellularAutomata(rule, grid, K);
            
        end
        % Gen ops
        function [child1, child2] = crossover(parent1, parent2)
            dim = rand < 0.5;
            dims = size(parent1.rule);

            % horizontal crossover
            if (dim)
                rows = dims(1);
                row = ceil(rand * (rows-1));
                child1 = [parent1.rule(1:row,:);parent2.rule(row+1:rows,:)];
                child2 = [parent2.rule(1:row,:);parent1.rule(row+1:rows,:)];

            % vertical crossover
            else
                cols = dims(2);
                col = ceil(rand * (cols-1));
                child1 = [parent1.rule(:,1:col),parent2.rule(:,col+1:cols)];
                child2 = [parent2.rule(:,1:col),parent1.rule(:,col+1:cols)];
            end
            
            % Wrap child rules in otca objects
            child1 = OuterTotalisticCellularAutomata(child1, parent1.grid, parent1.K);
            child2 = OuterTotalisticCellularAutomata(child2, parent2.grid, parent2.K);
            
        end
        function child = mutate(parent, mutation_rate)
            mutations = rand(size(parent.rule)) < mutation_rate;
            signs = sign(rand(size(parent.rule)) - 0.5);
            amount = floor(abs(normrnd(0,1,size(parent.rule))));
            child = parent.rule + (mutations .* signs .* amount);
            child = min(max(child, 0), parent.K); % force to legal states
            child = OuterTotalisticCellularAutomata(child, parent.grid, parent.K); % wrap
        end
    end
end