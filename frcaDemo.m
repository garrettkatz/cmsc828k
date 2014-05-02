% Cleanup variables and figures
clear;
close;

% Generate the Mackey glass time series with a random initial point
T = makeMackeyGlass(0.5+rand,17,0.1,50000);
T = T(10001:10:end); % subsample
T = tanh(T-1); % squash into (-1,1)
X = zeros(size(T));

% Use a 20 by 20 grid and 2 states
dims = [4 128]; % grid dimensions
K = 4;

% Construct random cellular automata
frca = FullRuleCellularAutomata.random(dims,K);
% Control lambda
lambda = 0.3;
frca.rule(rand(size(frca.rule)) > lambda) = 0;
frca.rule(1) = 0; % quiescent stays quiescent

% Wrap the cellular automata in a reservoir computer object for training
% (initialize the readout matrix to all zeros)
ext = randperm(numel(frca.a), 20);
readIn = 0; % no input
readOut = zeros(1, numel(frca.a));
readBack = 1; % feedback
rc = ReservoirComputer(frca, readIn, readOut, readBack);

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
err = (Y(:,t)-T(:,t)).^2;
err = mean(err(:));
1/err

% Plot the results, play movie of cellular automata
if false
    mx = max(A(:));
    step = 1;
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
        pause(1/48); % ~seconds per frame
    end
end