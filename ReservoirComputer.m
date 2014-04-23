classdef ReservoirComputer < handle
% A reservoir computer object is a wrapper which stores a handle to a
% reservoir and takes care of training.  A reservoir can be any object
% which satisfies the following contract:
%   -It has a field 'a'which is a column vector of unit activations
%   -It has a method with the signature 'pulse(x,y)' which updates the
%    activations based on input vector x and feedback vector y.
    properties
        reservoir; % Reservoir interface
        readIn; readOut; readBack; % input, output, feedback connection matrices
        y; % current output vector
    end
    methods
        function rc = ReservoirComputer(reservoir, readIn, readOut, readBack)
        % Reservoir computer constructs a new reservoir computer object
        % with the given reservoir and read in/out/back matrices.
        
            rc.reservoir = reservoir;
            rc.readIn = readIn;
            rc.readOut = readOut;
            rc.readBack = readBack;
            rc.y = zeros(size(readOut,1),1);
            
        end
        function reset(rc)
        % Reset resets the reservoir and output activations to zero.
        
            rc.reservoir.a(:) = 0;
            rc.y(:) = 0;
        
        end
        function [A,Y] = stream(rc, X, T)
        % Stream streams an input signal and optional target signal through
        % the reservoir.
        % Parameters:
        %   X : X(:,t) is the input vector at time t.
        %   T : T(:,t) is the target output vector at time t.  If provided,
        %       the actual output will be overwritten by this vector before
        %       feedback.
        % Returns:
        %   A : A(:,t) is the reservoir activation vector at time t.
        %   Y : Y(:,t) is the actual output vector at time t.
        
            % Pre-allocate records
            A = zeros(size(rc.reservoir.a,1), size(X,2));
            Y = zeros(size(rc.readOut,1), size(X,2));
            % Stream inputs
            for t = 1:size(X,2)
                % Record activations
                A(:,t) = rc.reservoir.a;
                Y(:,t) = rc.y;
                % Force output if given
                if nargin==3, rc.y = T(:,t); end;
                % Update
                x = rc.readIn*X(:,t);
                b = rc.readBack*rc.y;
                rc.reservoir.pulse(x, b);
                rc.y = rc.readOut*A(:,t);
            end
            
        end
        function err = train(rc, X, T, t, ridge)
        % Train optimizes the readout matrix to produce the desired target
        % signals from the given input signals.
        % Parameters:
        %   X : a cell array of input signals
        %   T : a cell array of target output signals
        %   t : a cell array of vectors of time steps.  the full signals
        %       are streamed, but only the time steps listed in t are used
        %       for regression. T{i}(:,t{i}+1) must be valid indexing.
        %   ridge : an optional ridge parameter for regression.  Defaults
        %       to zero(ordinary least squares regression).
        % Returns:
        %   err : training error on time-steps in t
        % If X, T, and t are singletons they do not need to be wrapped in
        % cell arrays.
        
            % Parameter handling
            if nargin < 5, ridge = 0; end;
            if ~iscell(X), X = {X}; end;
            if ~iscell(T), T = {T}; end;
            if ~iscell(t), t = {t}; end;
            
            % Generate training data
            A = cell(size(X));
            for i = 1:numel(X)
                rc.reset();
                [A{i}, ~] = rc.stream(X{i}, T{i});
                A{i} = A{i}(:,t{i});
                T{i} = T{i}(:,t{i}+1);
            end
            A = cat(2, A{:});
            T = cat(2, T{:});
            
            % Ridge regression on readout
            [U,S,V] = svd(A,'econ');
            S = diag(S);
            D = diag(S./(S.^2+ridge));
            rc.readOut = T*V*D*U';
            
            % Mean squared training error
            err = (T - rc.readOut*A).^2;
            err = mean(err(:));
            
        end
    end
end