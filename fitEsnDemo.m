% This file evaluates fitness of an ESN on mackey-glass and speech data.

% Cleanup variables and figures
clear;
close;

% Initialize an echo state network as in Jaeger 2010
W = 0.4*sign(sprandn(400, 400, 0.0125)); % Reservoir connections
esn = EchoStateNetwork(W);

% Wrap the echo state network in a reservoir computer objects
% Mackey
readIn = 0.14*sign(sprandn(400,1,0.5)); % Input connections
readOut = zeros(1,400);
readBack = 0.56*(2*rand(400,1)-1); % Feedback connections
rcMackey = ReservoirComputer(esn, readIn, readOut, readBack);
% Speech
readIn = 0.14*sign(sprandn(400,13,0.5)); % Input connections (13 channels)
readOut = zeros(10,400);
readBack = zeros(400,10); % No feedback (10 channels)
rcSpeech = ReservoirComputer(esn, readIn, readOut, readBack);

% Evaluate on tasks separately
[trainErr, testErr, genErr] = Fitness.evalMackey(rcMackey,true) % "true" means plot results
load SpeechData.mat % loads inputs Xsp and targets Tsp
[trainLoss, testLoss, genLoss] = Fitness.evalSpeech(rcSpeech, Xsp, Tsp)

% Aggregate performance on both tasks
fit = Fitness.eval(rcMackey, rcSpeech, Xsp, Tsp)
