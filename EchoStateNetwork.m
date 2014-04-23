classdef EchoStateNetwork < handle
% An echo state network object is an implementation of the reservoir
% contract.
    properties
        W; a; % Connection weight matrix and neuron activations.
        nu; % Noise parameter
        dC; dCa; % Leaky integration parameters
    end
    methods
        function esn = EchoStateNetwork(W, nu, dC, dCa)
        % Echo state network constructs a new echo state network object
        % with the given weight matrix, noise parameter, and leaky
        % integration parameters.
        
            % Parameter defaults
            if nargin < 4, dCa = 0.44*0.9; end;
            if nargin < 3, dC = 0.44; end;
            if nargin < 2, nu = 0.00001; end;
            
            % Set fields
            esn.W = W;
            esn.a = zeros(size(W,1),1);
            esn.nu = nu;
            esn.dC = dC;
            esn.dCa = dCa;
            
        end
        function pulse(esn, x, y)
        % Pulse updates activations based on input vector x and feedback
        % vector y.
        
            in = esn.W*esn.a + x + y; % Propogate signals
            in = in + esn.nu*(2*rand(size(in)) - 1); % add noise
            esn.a = (1-esn.dCa)*esn.a + esn.dC*tanh(in); % leak
            
        end
    end
end