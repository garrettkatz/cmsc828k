classdef CellularAutomata < handle
% A cellular automata object is an implementation of the reservoir
% contract.  Transition rules are represented by a function handle with the
% signature:
%   b = rule(g,a,x,y)
%   Parameters:
%       g : the adjacency matrix of the grid
%       a : the column vector of current activations
%       x : the input vector
%       y : the feedback vector
%   Returns:
%       b : the column vector of new activations after transition
    properties
        rule; % Rule function handle
        grid; % Grid adjacency matrix
        K; % Number of states(excluding quiescent state zero)
        a; % Column vector of unit activations
    end
    methods
        function ca = CellularAutomata(rule, grid, K)
        % Cellular automata constructs a cellular automata object with the
        % given rule, grid, and number of states.
        
            ca.rule = rule;
            ca.grid = grid;
            ca.K = K;
            ca.a = zeros(size(grid,1),1);
            
        end
        function pulse(ca, x, y)
        % Pulse updates activations based on input vector x and feedback
        % vector y.
        
            % Apply rule
            ca.a = ca.rule(ca.grid, ca.a, x, y);
            % Enforce valid states
            ca.a = CellularAutomata.force(ca.a, ca.K);
            
        end
    end
    methods(Static = true)
        function w = force(v,K)
        % Force rounds and truncates an activation vector to ensure it contains only legal states
        % Parameters:
        %   v : a column vector of(potentially invalid) activations
        %   K : Number of states(excluding quiescent state zero)
        % Returns:
        %   w : the rounded, truncated version of v
        
            w = min(max(round(v),0),K);
            
        end
        function grid = makeGrid(dims)
        % Make grid makes the adjacency matrix for an n-dimensional grid
        % with non periodic boundary conditions and the given dimensions
        % using the von Neumann (5) neighborhood.
        % Parameters:
        %   dims : dims(i) is the grid size in the i^th dimension
        % Returns:
        %   grid: the (sparse) adjacency matrix for the grid.
        
            if numel(dims)==0 % Base case
                grid = sparse(1,1,1);
            else % Recurse
                
                % Build the grid for previous dimensions
                subgrid = CellularAutomata.makeGrid(dims(1:end-1));
                
                % Replicate along the main diagonal
                blocks = repmat({subgrid},dims(end),1);
                grid = blkdiag(blocks{:});
                
                % Add off-diagonals for this dimension
                newEntries = ones(prod(dims),2);
                newDiags = [-1 1]*prod(dims(1:end-1));
                grid = spdiags(newEntries, newDiags, grid);
                
            end
        end
    end
end