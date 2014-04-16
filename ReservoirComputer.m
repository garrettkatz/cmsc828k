classdef ReservoirComputer < handle
% A reservoir computer object is a wrapper which stores a handle to a
% reservoir and takes care of training.  A reservoir can be any object
% which satisfies the following contract:
%   -It has a field 'a'which is a column vector of unit activations
%   -It has a method with the signature 'pulse(x,y)' which updates the
%    activations based on input vector x and feedback vector y.
    properties
        reservoir; % Reservoir interface
        readOut; % Read-out matrix
        y; % Output vector
    end
    methods
        function rc = ReservoirComputer(reservoir, readOut)
        % Reservoir computer constructs a new reservoir computer object
        % with the given reservoir and readout matrix.
        
            rc.reservoir = reservoir;
            rc.readOut = readOut;
            rc.y = zeros(size(readOut,1),1);
            
        end
        function reset(rc)
        % Reset resets the reservoir and output activations to zero.
        
            rc.reservoir.a(:) = 0;
            rc.y(:) = 0;
        
        end
        function [A,Y] = stream(rc, X, T)
        % Stream streams an input signal and optional target signal through the reservoir.
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
                rc.reservoir.pulse(X(:,t), rc.y);
                rc.y = rc.readOut*A(:,t);
            end
            
        end
        function train(rc, X, T, t, ridge)
        % Train optimizes the readout matrix to produce the desired target
        % signal from the given input signal.
        % Parameters:
        %   X : the input signal
        %   T : the target output signal
        %   t : a vector of time steps.  the full signals are streamed, but
        %       only the time steps listed in t are used for regression.
        %   ridge : an optional ridge parameter for regression.  Defaults
        %       to zero(ordinary least squares regression).
        
            if nargin < 5, ridge = 0; end;
            
            % Generate training data
            [A,~] = rc.stream(X, T);
            
            % Ridge regression on readout
            [U,S,V] = svd(A(:,t),'econ');
            S = diag(S);
            D = diag(S./(S.^2+ridge));
            rc.readOut = T(:,t+1)*V*D*U';
            
        end
    end
end