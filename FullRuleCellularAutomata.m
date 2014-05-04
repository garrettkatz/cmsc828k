classdef FullRuleCellularAutomata < handle
% A FullRuleCellularAutomata object stores the full rule table for every
% possible neighborhood configuration.  Assumes 5-neighborhood, 2d CA with
% periodic boundaries and binary states.  The i^th neighborhood
% configuration is the base-K number
%       i = K^4*self + K^3*left + K^2*right + K*up + down
    properties
        rule; % rule(i) is the new state for i^th neighborhood configuration
        neighbors; % N x 5 neighbor indices (including self)
        K; % number of states
        a; % column vector of unit activations
        readOut;
        cts;
    end
    methods
        function frca = FullRuleCellularAutomata(rule, neighbors, K, cts)
        % FullRuleCellularAutomata constructs a cellular automata
        % object with the given rule table and dimensions.
        
            if nargin > 0 % check for object array construction
                frca.rule = rule;
                frca.neighbors = neighbors;
                frca.K = K;
                frca.a = zeros(size(neighbors,1), 1);
                if nargin < 4, cts = 0; end;
                frca.cts = cts;
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
            cts = frca.cts;
            
            % Feedback (Map to node index)
            a(ceil(N*(tanh(b)+1)/2)) = K-1;
            % Get neighborhood states
            neighborhood = round(reshape(a(neighbors(:)),N,5));
            % Apply rules
            a = cts*a + (1-cts)*rule(neighborhood*pows+1);
            
            frca.a = a;
        end
        function [fit, Y] = fitness(frca, X, T)
        % Inline fitness evaluation of otcas on mackey-glass

            % Localize variables
            a = zeros(size(frca.a));
            N = numel(a);
            rule = frca.rule;
            neighbors = frca.neighbors;
            K = frca.K;
            pows = K.^(0:4)'; % conversion to base K
            len = size(T,2);
            cts = frca.cts;

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
                neighborhood = round(reshape(a(neighbors(:)),N,5));
                % Apply rules
                a = cts*a + (1-cts)*rule(neighborhood*pows+1);
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
                neighborhood = round(reshape(a(neighbors(:)),N,5));
                % Apply rules
                a = cts*a + (1-cts)*rule(neighborhood*pows+1);
                % Output
                y = readOut*A(:,t);
            end
            
            % Save readout
            frca.readOut = readOut;
            
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
            clone.cts = frca.cts;
        end
        function check(frca, X, T, dims, step, framerate)
        % check visualizes performance on Mackey-glass
        
             % Localize variables
            a = zeros(size(frca.a));
            N = numel(a);
            rule = frca.rule;
            neighbors = frca.neighbors;
            K = frca.K;
            pows = K.^(0:4)'; % conversion to base K
            len = size(T,2);
            readOut = frca.readOut;
            cts = frca.cts;
            
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
                neighborhood = round(reshape(a(neighbors(:)),N,5));
                % Apply rules
                a = cts*a + (1-cts)*rule(neighborhood*pows+1);
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
        function frca = random(dims,K,lambda,cts)
        % random constructs a randomized FullRuleCellularAutomata.
        
            rule = randi(K, [K^5, 1])-1;
            neighbors = FullRuleCellularAutomata.makeNeighbors(dims);
            if nargin < 4, cts = rand; end;
            frca = FullRuleCellularAutomata(rule, neighbors, K, cts);
            frca.rule(1) = 0; % quiescent stays quiescent
            % Control lambda
            frca.rule(rand(size(frca.rule)) > lambda) = 0;

        end
        % Gen ops
        function [child1, child2] = crossover(parent1, parent2)
            % Cross-over rule tables, assuming equal K
            cutpt = randi(parent1.K-1);
            rule1 = [parent1.rule(1:cutpt); parent2.rule(cutpt+1:end)];
            rule2 = [parent2.rule(1:cutpt); parent1.rule(cutpt+1:end)];
            
            % intermediate cts
            cts = (parent1.cts+parent2.cts)/2;
            
            % Wrap child rules in frca objects
            child1 = FullRuleCellularAutomata(rule1, parent1.neighbors, parent1.K, cts);
            child2 = FullRuleCellularAutomata(rule2, parent2.neighbors, parent2.K, cts);
            
        end
        function child = mutate(parent, mutation_rate)
            rule = parent.rule;
            mutations = rand(size(parent.rule)) < mutation_rate;
            new_rules = rule(mutations) + randn(nnz(mutations), 1);
            rule(mutations) = max(min(round(new_rules), parent.K-1),0);
            cts = mutation_rate*rand + (1-mutation_rate)*parent.cts;
            child = FullRuleCellularAutomata(rule, parent.neighbors, parent.K, cts); % wrap
        end
    end
end