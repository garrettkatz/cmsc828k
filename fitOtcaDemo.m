% This file evaluates fitness of an ESN on mackey-glass and speech data.

% Cleanup variables and figures
clear;
close;

% Use a 20 by 20 grid and 256 states
dims = [20 20]; % grid dimensions
K = 256;

% Construct random cellular automata
otca = OuterTotalisticCellularAutomata.random(dims,K);

% Change rule to "move toward neighborhood average"
states = repmat((0:K)', 1, 6*K+1); % 6 to include input/feedback
sums = repmat(0:6*K, K+1, 1);
rule = states - (states > (states+sums)/5) + (states < (states+sums)/5);
otca.rule = min(rule,K);

% Wrap the cellular automata in reservoir computer objects
N = numel(otca.a); % number of units
% Mackey
ext = randperm(N, 20); % indices of external-signal-receiving units
readIn = sparse(ext(1:10), 1, 1, N, 1); % 1st 10 for input
readOut = zeros(1, N);
readBack = sparse(ext(11:20), 1, 1, N, 1); % last 10 for feedback
rcMackey = ReservoirComputer(otca, readIn, readOut, readBack);
% Speech
numIn = 2; % Number of input-receiving units per channel
ext = randperm(N, numIn*13); % 13 input channels
readIn = sparse(ext, repmat(1:13, 1, numIn), 1, N, 13);
readOut = zeros(10, N); % 10 output channels
readBack = zeros(N, 10); % no feedback
rcSpeech = ReservoirComputer(otca, readIn, readOut, readBack);

% Evaluate on tasks separately
[trainErr, testErr, genErr] = Fitness.evalMackey(rcMackey,true) % "true" means plot results
load SpeechData.mat % loads inputs Xsp and targets Tsp
[trainLoss, testLoss, genLoss] = Fitness.evalSpeech(rcSpeech, Xsp, Tsp)

% Aggregate performance on both tasks
fit = Fitness.eval(rcMackey, rcSpeech, Xsp, Tsp)
