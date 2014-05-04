classdef OuterTotalisticCellularAutomata < handle
% An OuterTotalisticCellularAutomata object is an implementation of the
% reservoir contract.  Transition rules are represented by a table where
% entry (k+1, nsum+1) contains the new state for a unit with old state "k"
% and old neighborhood sum "nsum".  The grid is represented by an adjacency
% matrix "grid", so that grid*a is a vector of neighborhood sums.
    properties
        rule; % Rule table
        grid; % Grid adjacency matrix
        K; % Number of states (excluding quiescent state zero)
        a; % Column vector of unit activations
        readIn; readBack; readOut; % reservoir matrices
        cts; % continuity parameter
    end
    methods
        function otca = OuterTotalisticCellularAutomata(rule, grid, K, cts)
        % OuterTotalisticCellularAutomata constructs a cellular automata
        % object with the given rule, grid, and number of states.
        
            % Connections (persistent to avoid confounding factors)
            persistent readIn;
            persistent readBack;
            if isempty(readIn)
                N = size(grid,1);
                ext = randperm(N, 20); % indices of external-signal-receiving units
                readIn = sparse(ext(1:10), 1, 1, N, 1); % 1st 10 for input
                readBack = sparse(ext(11:20), 1, 1, N, 1); % last 10 for feedback
            end
        
            if nargin > 0 % check for object array construction
                otca.rule = rule;
                otca.grid = grid;
                otca.K = K;
                otca.a = zeros(size(grid,1), 1);
                otca.readIn = readIn;
                otca.readBack = readBack;
                if nargin < 4, cts = 0; end;
                otca.cts = cts;
            end
            
        end
        function pulse(otca, x, b)
        % Pulse updates activations based on input vector x and feedback
        % vector b.

            K = otca.K;
            a = otca.a;
            cts = otca.cts;

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
            idx = round(a) + (K+1)*nsum + 1;

            % Apply rule
            otca.a = cts*a + (1-cts)*otca.rule(idx);
           
%             % MEX attempt (buggy)
%             nsum = otca.grid*otca.a;
%             new_a = otcapulse(nsum, otca.rule, otca.K, x, b, otca.a);
%             otca.a = new_a;
            
        end
        function [fit, Y] = fitness(otca, X, T)
        % Inline fitness evaluation of otcas on mackey-glass

            % Localize variables
            a = zeros(size(otca.a));
            N = numel(a);
            grid = otca.grid;
            rule = otca.rule;
            K = otca.K;
            len = size(T,2);
            readIn = otca.readIn;
            readBack = otca.readBack;
            cts = otca.cts;
            
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
                idx = round(a) + (K+1)*nsum + 1;
                % Apply rule
                a = cts*a + (1-cts)*rule(idx);
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
                % Input and feedback
                x = readIn*X(:,t);
                b = readBack*y;
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
                idx = round(a) + (K+1)*nsum + 1;
                % Apply rule
                a = cts*a + (1-cts)*rule(idx);
                % Output
                y = readOut*A(:,t);
            end
            
            % Save readOut
            otca.readOut = readOut;
            
            % Mean squared testing error
            err = (T - Y).^2;
            err = mean(err(:));
            fit = 1/err;
            
        end
        function clone = copy(otca)
            clone = OuterTotalisticCellularAutomata(otca.rule, otca.grid, otca.K, otca.cts);
        end
        function check(otca, X, T, dims, step, framerate)
        % check visualizes performance on Mackey-glass
            
            % Localize variables
            a = zeros(size(otca.a));
            N = numel(a);
            grid = otca.grid;
            rule = otca.rule;
            K = otca.K;
            len = size(T,2);
            readIn = otca.readIn;
            readBack = otca.readBack;
            readOut = otca.readOut;
            cts = otca.cts;
            
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
                % Input and feedback
                x = readIn*X(:,t);
                b = readBack*y;
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
                idx = round(a) + (K+1)*nsum + 1;
                % Apply rule
                a = cts*a + (1-cts)*rule(idx);
                % Output
                y = readOut*A(:,t);
            end

            mx = max(A(:));
            for t = 1:step:size(A,2)
                subplot(3,1,1);
                plot(1:size(A,2),T,'b',1:size(A,2),Y,'r',[t t],[-1 1],'k');
                title('target output vs actual');
                legend('target','actual');
                xlabel('time');
                ylabel('output activation');
                subplot(3,1,2);
                plot(1:size(A,2),A',[t t],[0 mx],'k');
                title('reservoir (plot)')
                xlabel('time');
                ylabel('unit activation');
                subplot(3,1,3);
                imshow(reshape(A(:,t),[dims(1),prod(dims(2:end))])/mx);
                title('reservoir over time (brightness = activation)')
                pause(framerate); % ~seconds per frame
            end
        end
    end
    methods(Static = true)
        function grid = makeGrid(dims)
        % Make grid makes the adjacency matrix for an n-dimensional grid
        % with non periodic boundary conditions and the given dimensions
        % using the von Neumann (5) neighborhood.
        % Parameters:
        %   dims : dims(i) is the grid size in the i^th dimension
        % Returns:
        %   grid: the (sparse) adjacency matrix for the grid.
        
            if numel(dims)==0 % Base case

                grid = sparse(1,1,0); % No self-connections
                
            else % Recurse
                
                % Build the grid for previous dimensions
                subgrid = OuterTotalisticCellularAutomata.makeGrid(dims(1:end-1));
                
                % Replicate along the main diagonal
                blocks = repmat({subgrid},dims(end),1);
                grid = blkdiag(blocks{:});
                
                % Add off-diagonals for this dimension
                newEntries = ones(prod(dims),2);
                newDiags = [-1 1]*prod(dims(1:end-1));
                grid = spdiags(newEntries, newDiags, grid);
                
            end
        end
        function grid = makeGridPeriodic(dims)
            if (numel(dims) ~= 2)
                grid = [];
                return
            end
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
            grid = zeros(prod(dims),prod(dims));
            for i=1:size(neighbors,1)
                grid(i,neighbors(i,:)) = 1;
            end
            grid = sparse(grid);
        end
        function otca = random(dims, K, cts)
        % random constructs a randomized OuterTotalisticCellularAutomata.
        
            persistent grid
            if (isempty(grid))
                grid = OuterTotalisticCellularAutomata.makeGridPeriodic(dims);
            end
        
            rule = randi(K+1, [K+1, 6*K+1])-1; % 6 to include input/feedback
            
            if nargin < 3, cts = 0; end;
            otca = OuterTotalisticCellularAutomata(rule, grid, K, cts);
            
        end
        function otca = smooth(dims, K, cts)
        % smooth constructs a smooth-rule OuterTotalisticCellularAutomata.
        
            rule = randi(K+1, [K+1, 6*K+1])-1; % 6 to include input/feedback
            grid = OuterTotalisticCellularAutomata.makeGrid(dims);
            if nargin < 3, cts = 0.9; end;
            otca = OuterTotalisticCellularAutomata(rule, grid, K, cts);
            
            % Change rule to "move toward neighborhood average"
            states = repmat((0:K)', 1, 6*K+1); % 6 to include input/feedback
            sums = repmat(0:6*K, K+1, 1);
            %rule = states - (states > (states+sums)/5) + (states < (states+sums)/5);
            rule = (states+sums)/5;
            otca.rule = min(max(rule,0),K);

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
            child1 = OuterTotalisticCellularAutomata(child1, parent1.grid, parent1.K, parent1.cts);
            child2 = OuterTotalisticCellularAutomata(child2, parent2.grid, parent2.K, parent2.cts);
            
        end
        function child = mutate(parent, mutation_rate)
            mutations = rand(size(parent.rule)) < mutation_rate;
            signs = sign(rand(size(parent.rule)) - 0.5);
            amount = floor(abs(normrnd(0,1,size(parent.rule))));
            child = parent.rule + (mutations .* signs .* amount);
            child = min(max(child, 0), parent.K); % force to legal states
            child = OuterTotalisticCellularAutomata(child, parent.grid, parent.K, parent.cts); % wrap
        end
    end
end