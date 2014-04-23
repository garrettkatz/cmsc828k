% This file uses the reservoir computing learning technique to train a
% cellular automata on the Mackey glass time series.  I've tried several
% transition rules, some of which work better than others.  None of them
% are very effective, but hey, that's what evolution is for :).  You can
% uncomment the different rules below to try them out.  You will probably
% have to run this several times to see some interesting behavior.  Each
% time it will spend awhile playing a movie of the activations, so just
% press control -c to stop early if it's boring (e.g. fixed point).

% Cleanup variables and figures
clear;
close;

% Generate the Mackey glass time series with a random initial point
T = makeMackeyGlass(0.5+rand,17,0.1,50000);
T = T(10001:10:end); % subsample
T = tanh(T-1); % squash into (-1,1)
X = 0.2*ones(size(T)); % constant bias
%X = zeros(size(T)); % constant bias

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

% Wrap the cellular automata in a reservoir computer object for training
% (initialize the readout matrix to all zeros)
ext = randperm(numel(otca.a), 20);
readIn = sparse(ext(1:10), 1, 1, numel(otca.a), 1);
readOut = zeros(1, numel(otca.a));
readBack = sparse(ext(11:20), 1, 1, numel(otca.a), 1);
rc = ReservoirComputer(otca, readIn, readOut, readBack);

% Train on the Mackey glass data (only regress on middle 2000)
t = 1000:3000;
rc.train(X, T, t, 10);

% Reset the reservoir to zero activations and stream the first 2000 time
% steps with "teacher forcing" (target output is used for feedback)
rc.reset();
t = 1:2000;
[A(:,t),Y(:,t)] = rc.stream(X(:,t),T(:,t));

% Stream the remainder without teacher forcing
t = 2001:4000;
[A(:,t),Y(:,t)] = rc.stream(X(:,t));

% Plot the results, play movie of cellular automata
mx = max(A(:));
for t = 1:10:size(A,2)
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
    pause(1/48); % ~seconds per frame
end
