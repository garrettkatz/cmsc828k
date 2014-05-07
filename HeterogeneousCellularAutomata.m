classdef HeterogeneousCellularAutomata < handle
% A HeterogenousCellularAutomata object is like FullRuleCellularAutomata,
% except a single CA can have different rule sets for different nodes.
    properties
        rule; % rule(i,j) is new state for unit j with neighborhood configuration i
        neighbors; % N x 5 neighbor indices (including self)
        K; % number of states
        a; % column vector of unit activations
        readOut; % readout connections
        cts;
        lambda;
    end
    methods
        function hca = HeterogeneousCellularAutomata(rule, neighbors, K, cts)
        % HeterogeneousCellularAutomata constructs a cellular automata
        % object with the given rule table and dimensions.
        
            if nargin > 0 % check for object array construction
                hca.rule = rule;
                hca.neighbors = neighbors;
                hca.K = K;
                hca.a = zeros(size(neighbors,1), 1);
                if nargin < 4, cts = 0; end;
                hca.cts = cts;
                hca.lambda = 1-(nnz(rule)/numel(rule));
            end
            
        end
        function pulse(hca, x, b)
            
            % Localize variables
            a = hca.a;
            N = numel(a);
            rule = hca.rule;
            neighbors = hca.neighbors;
            K = hca.K;
            pows = K.^(0:4)'; % conversion to base K
            cts = hca.cts;
            
            % Feedback (Map to node index)
            a(ceil(N*(tanh(b)+1)/2)) = K-1;
            % Get neighborhood states
            neighborhoods = round(reshape(a(neighbors(:)),N,5));
            % Apply rule
            idx = neighborhoods*pows + (K^5)*(0:numel(a)-1)' + 1; 
            a = cts*a + (1-cts)*rule(idx);
            hca.a = a;
        end
        function [fit, Y] = fitness(hca, X, T)
        % Inline fitness evaluation of otcas on mackey-glass

            % Localize variables
            a = zeros(size(hca.a));
            N = numel(a);
            rule = hca.rule;
            neighbors = hca.neighbors;
            K = hca.K;
            pows = K.^(0:4)'; % conversion to base K
            cts = hca.cts;
            len = size(T,2);

            % Generate training data
            % Pre-allocate records
            A = zeros(N, len);
            % Stream inputs
            for t = 1:len
                % Record activations
                A(:,t) = a;
                % Input + Feedback (Map to node index)
                a(ceil(N*(tanh(X(:,t))+1)/2)) = K-1;
                a(ceil(N*(tanh(T(:,t))+1)/2)) = K-1;
                % Get neighborhood states
                neighborhoods = round(reshape(a(neighbors(:)),N,5));
                % Apply rule
                idx = neighborhoods*pows + (K^5)*(0:numel(a)-1)' + 1;
                a = cts*a + (1-cts)*rule(idx); % A(:,t+1) = fun( A(:,t), T(:,t) )
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
                % Input + Feedback (Map to node index)
                a(ceil(N*(tanh(X(:,t))+1)/2)) = K-1;
                a(ceil(N*(tanh(y)+1)/2)) = K-1;
                % Get neighborhood states
                neighborhoods = round(reshape(a(neighbors(:)),N,5));
                % Apply rule
                idx = neighborhoods*pows + (K^5)*(0:numel(a)-1)' + 1; 
                a = cts*a + (1-cts)*rule(idx); % A(:,t+1) = fun( A(:,t), T(:,t) )
                % Output
                y = readOut*A(:,t); % Y(:,t+1) = fun( A(:,t) )
            end
            
            % Save readout
            hca.readOut = readOut;
            
            % Mean squared testing error
            err = (T - Y).^2;
            err = mean(err(:));
            fit = 1/err;
        end
        function clone = copy(hca)
            clone = HeterogeneousCellularAutomata;
            clone.rule = hca.rule;
            clone.neighbors = hca.neighbors;
            clone.K = hca.K;
            clone.a = hca.a;
            clone.cts = hca.cts;
        end
        function check(hca, X, T, dims, step, framerate)
        % check visualizes performance on Mackey-glass
        
            % Localize variables
            a = zeros(size(hca.a));
            N = numel(a);
            rule = hca.rule;
            neighbors = hca.neighbors;
            K = hca.K;
            pows = K.^(0:4)'; % conversion to base K
            len = size(T,2);
            readOut = hca.readOut;
            cts = hca.cts;

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
                % Input + Feedback (Map to node index)
                a(ceil(N*(tanh(X(:,t))+1)/2)) = K-1;
                a(ceil(N*(tanh(y)+1)/2)) = K-1;
                % Get neighborhood states
                neighborhoods = round(reshape(a(neighbors(:)),N,5));
                % Apply rule
                idx = neighborhoods*pows + (K^5)*(0:numel(a)-1)' + 1; 
                a = cts*a + (1-cts)*rule(idx); % A(:,t+1) = fun( A(:,t), T(:,t) )
                % Output
                y = readOut*A(:,t); % Y(:,t+1) = fun( A(:,t) )
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
        function hca = random(dims, K, lambda, cts)
        % random constructs a randomized HeterogeneousCellularAutomata.
        
            rule = randi(K, [K^5, prod(dims)])-1;
            neighbors = HeterogeneousCellularAutomata.makeNeighbors(dims);
            if nargin < 4, cts = rand; end;
            hca = HeterogeneousCellularAutomata(rule, neighbors, K, cts);
            hca.rule(1,:) = 0; % quiescent stays quiescent
            % Control lambda
            hca.rule(rand(size(hca.rule)) > lambda) = 0;

        end
        function hca = gauss(dims, K, cts)
        % gauss constructs a gauss-rule HeterogeneousCellularAutomata.
        
            persistent neighbors;
            if isempty(neighbors)
                neighbors = HeterogeneousCellularAutomata.makeNeighbors(dims);
            end
            if nargin < 3, cts = rand; end;
            num_guass = 64;
            rule = K/2*ones([K^5, prod(dims)]);
            for n = 1:num_guass
                rule = rule + OuterTotalisticCellularAutomata.gaussRule(size(rule), K/2, min(size(rule))/25);
            end
            rule = min(max(round(rule),0),K-1);

            hca = HeterogeneousCellularAutomata(rule, neighbors, K, cts);

        end
        % Gen ops
        function [child1, child2] = smoothCrossover(parent1, parent2)
            
            sr = OuterTotalisticCellularAutomata.sigRule(size(parent1.rule));
            rule1 = sr.*parent1.rule + (1-sr).*parent2.rule;
            rule1 = min(max(round(rule1),0),parent1.K-1);
            rule2 = sr.*parent2.rule + (1-sr).*parent1.rule;
            rule2 = min(max(round(rule2),0),parent1.K-1);
            
            % Wrap child rule in otca objects
            child1 = HeterogeneousCellularAutomata(rule1, parent1.neighbors, parent1.K, parent1.cts);
            child2 = HeterogeneousCellularAutomata(rule2, parent2.neighbors, parent2.K, parent2.cts);
            
        end
        function child = gaussMutate(parent, mutation_rate, mutate_cts)
            
            % Mutate rule with gaussians
            num_mutators = 10;
            rule = parent.rule;
            for m = 1:num_mutators
                rule = rule + OuterTotalisticCellularAutomata.gaussRule(size(rule), parent.K/2*mutation_rate, min(size(rule))/5*mutation_rate);
            end
            rule = min(max(round(rule),0),parent.K-1);
            
            % Mutate continuity if requested (default yes)
            cts = parent.cts;
            if nargin < 3, mutate_cts = true; end;
            if mutate_cts
                cts = mutation_rate*rand + (1-mutation_rate)*cts;
            end
            
            child = HeterogeneousCellularAutomata(rule, parent.neighbors, parent.K, cts); % wrap
            
        end
        function [child1, child2] = crossover(parent1, parent2)
            % Cross-over units
            cutpt = randi(numel(parent1.a)-1);
            rule1 = [parent1.rule(:,1:cutpt), parent2.rule(:,cutpt+1:end)];
            rule2 = [parent2.rule(:,1:cutpt), parent1.rule(:,cutpt+1:end)];

            % intermediate cts
            cts = (parent1.cts+parent2.cts)/2;
            
            % Wrap child rule in frca objects
            child1 = HeterogeneousCellularAutomata(rule1, parent1.neighbors, parent1.K, cts);
            child2 = HeterogeneousCellularAutomata(rule2, parent2.neighbors, parent2.K, cts);
            
        end
        function child = mutate(parent, mutation_rate)
            child = parent.copy();
            mutations = rand(size(parent.rule)) < mutation_rate;
            new_rule = child.rule(mutations) + randn(nnz(mutations), 1);
            child.rule(mutations) = max(min(round(new_rule), parent.K-1),0);
            child.cts = mutation_rate*rand + (1-mutation_rate)*parent.cts;
        end
    end
end