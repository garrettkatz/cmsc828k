classdef OuterTotalisticCellularAutomata < handle
% An OuterTotalisticCellularAutomata object is an implementation of the
% reservoir contract.  Transition rules are represented by a table where
% entry (k+1, nsum+1) contains the new state for a unit with old state "k"
% and old neighborhood sum "nsum".  The grid is represented by an adjacency
% matrix "grid", so that grid*a is a vector of neighborhood sums. xIn and
% yIn are indices of units that receive external input or feedback,
% respectively.
    properties
        rule; % Rule table
        grid; % Grid adjacency matrix
        K; % Number of states (excluding quiescent state zero)
        a; % Column vector of unit activations
        xIn; yIn; % weights for external signals
    end
    methods
        function otca = OuterTotalisticCellularAutomata(rule, grid, K, xIn, yIn)
        % OuterTotalisticCellularAutomata constructs a cellular automata
        % object with the given rule, grid, and number of states.
        
            otca.rule = rule;
            otca.grid = grid;
            otca.K = K;
            otca.a = zeros(size(grid,1), 1);
            otca.xIn = xIn; otca.yIn = yIn;
            
        end
        function pulse(otca, x, y)
        % Pulse updates activations based on input vector x and feedback
        % vector y.
        
            % Get neighborhood sum including external signals
            nsum = otca.grid*otca.a;
            nsum(otca.xIn) = nsum(otca.xIn) + x;
            nsum(otca.yIn) = nsum(otca.yIn) + y;
            
            % Force to indices
            nsum = min(max(round(nsum), 0), size(otca.rule,2));

            % Get linear indices in rule table
            %idx = [otca.a + 1, otca.grid*otca.a + 1]*[1; otca.K+1] - otca.K; % faster?
            idx = sub2ind(size(otca.rule), otca.a+1, nsum + 1);

            % Apply rule
            otca.a = otca.rule(idx);
            
        end
        function baby = recombine(otca, babyDaddy)
            % ?????????
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
        function otca = random(dims, K, xNum, yNum)
        % random constructs a randomized OuterTotalisticCellularAutomata.
            rule = randi(K+1, [K+1, 4*K+1])-1;
            grid = OuterTotalisticCellularAutomata.makeGrid(dims);
            in = randperm(prod(dims), xNum+yNum);
            xIn = in(1:xNum);
            yIn = in(xNum+1:end);
            otca = OuterTotalisticCellularAutomata(rule, grid,K, xIn, yIn);
        end
    end
end