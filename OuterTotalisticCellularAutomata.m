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
        lambda;
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
                otca.lambda = 1-(nnz(otca.rule)/numel(otca.rule));
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
%                 size(a)
%                 size(nsum)
%                 max(xi)
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
        function lambda = summary(otca)
            lambda = otca.lambda;
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
        function otca = random(dims, K, lambda, cts)
        % random constructs a randomized OuterTotalisticCellularAutomata.
        
            rule = randi(K, [K+1, 6*K+1]); % 6 to include input/feedback
            rule(rand(size(rule)) < lambda) = 0;
            grid = OuterTotalisticCellularAutomata.makeGrid(dims);
            if nargin < 4, cts = rand; end;
            otca = OuterTotalisticCellularAutomata(rule, grid, K, cts);
            
        end
        function otca = smooth(dims, K, lambda, cts)
        % smooth constructs a smooth-rule OuterTotalisticCellularAutomata.
        
            if nargin < 4, cts = rand; end;
            otca = OuterTotalisticCellularAutomata.random(dims,K,lambda,cts);
            
            % Change rule to "move toward neighborhood average"
            states = repmat((0:K)', 1, 6*K+1); % 6 to include input/feedback
            sums = repmat(0:6*K, K+1, 1);
            rule = (states+sums)/5;
            otca.rule = min(max(round(rule),0),K);
            otca.lambda = 1-(nnz(otca.rule)/numel(otca.rule));

        end
        function otca = gauss(dims, K, cts)
        % gauss constructs a gauss-rule OuterTotalisticCellularAutomata.
        
            if nargin < 3, cts = rand; end;
            otca = OuterTotalisticCellularAutomata.random(dims,K,0,cts);
            
            num_guass = 10;
            otca.rule(:) = K/2;
            for n = 1:num_guass
                otca.rule = otca.rule + OuterTotalisticCellularAutomata.gaussRule(size(otca.rule), K/2, K/5);
            end
            otca.rule = min(max(otca.rule,0),K);
            otca.lambda = 1-(nnz(otca.rule)/numel(otca.rule));

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
        function [child1, child2] = smoothCrossover(parent1, parent2)
            
            sr = OuterTotalisticCellularAutomata.sigRule(size(parent1.rule));
            rule1 = sr.*parent1.rule + (1-sr).*parent2.rule;
            rule1 = min(max(round(rule1),0),parent1.K);
            rule2 = sr.*parent2.rule + (1-sr).*parent1.rule;
            rule2 = min(max(round(rule2),0),parent1.K);
            
            % Wrap child rules in otca objects
            child1 = OuterTotalisticCellularAutomata(rule1, parent1.grid, parent1.K, parent1.cts);
            child2 = OuterTotalisticCellularAutomata(rule2, parent2.grid, parent2.K, parent2.cts);
            
        end
        function child = mutate(parent, mutation_rate)
            
            mutations = rand(size(parent.rule)) < mutation_rate;
            signs = sign(rand(size(parent.rule)) - 0.5);
            amount = floor(abs(parent.K/10*normrnd(0,1,size(parent.rule))));
            child = parent.rule + (mutations .* signs .* amount);
            child = min(max(round(child),0),parent.K);
            child = OuterTotalisticCellularAutomata(child, parent.grid, parent.K, parent.cts); % wrap
            
        end
        function rule = gaussRule(sz, height, width)
            persistent i;
            persistent j;
            if isempty(i)
                [i, j] = meshgrid(1:sz(1),1:sz(2));
                i = i';
                j = j';
            end
            i_c = randi(sz(1));
            j_c = randi(sz(2));
            rule = randn*height*exp(-((i-i_c).^2 + (j-j_c).^2)/(2*width)^2);
        end
        function rule = sigRule(sz, width)
            persistent i;
            persistent j;
            if isempty(i)
                [i, j] = meshgrid(1:sz(1),1:sz(2));
                i = i';
                j = j';
            end
            if nargin < 2, width = 0.1; end;
            i_c = randi(sz(1));
            j_c = randi(sz(2));
            rule = 1./(1+exp(-width*(i-i_c))).*1./((1+exp(-width*(j-j_c))));
        end
        function child = gaussMutate(parent, mutation_rate)
            
            num_mutators = 10;
            rule = parent.rule;
            for m = 1:num_mutators
                rule = rule + OuterTotalisticCellularAutomata.gaussRule(size(rule), parent.K/2*mutation_rate, parent.K/5*mutation_rate);
            end
            rule = min(max(round(rule),0),parent.K);
            child = OuterTotalisticCellularAutomata(rule, parent.grid, parent.K, parent.cts); % wrap
            
        end
    end
end