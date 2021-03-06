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
T = floor(128*(T+1)); % discretize to 256 states
X = 52*ones(size(T)); % constant bias (0.2*256)

% Use a 20 by 20 grid and 256 states
dims = [20 20]; % grid dimensions
grid = CellularAutomata.makeGrid(dims);
K = 256;
% Randomly select 1% of the units to receive external input
ins = rand(prod(dims),2)<0.01;

%%% Various rules

% Outer-totalistic rule with random lookup table
rtable = (rand(7*K+1,K+1)<0.3).*randi(K,[7*K+1,K+1]);
rule = @(g,a,x,y)...
    rtable(CellularAutomata.force(g*a - a + ins*[x;y], K) + a*(7*K+1) + 1);

% Excitable media rule
% if a is odd: a += 2*del
% if a is even: a -= 2*del
% if a is zero with odd neighbor: a = 1
del = 6;
rule = @(g,a,x,y) ...
    a - 2*del*(-1).^mod(a,2) + (2*del+1)*(a==0 & g*mod(a+ins*[x;y],2)>0);

% Arbitrary rule (approach neighbor average)
rule = @(g,a,x,y) a + (-1).^((ins*[x;y] + g*a)/5 <= a);

% Construct cellular automata
ca = CellularAutomata(rule, grid, K);

% Wrap the cellular automata in a reservoir computer object for training
% (initialize the readout matrix to all zeros)
rc = ReservoirComputer(ca, zeros(1, numel(ca.a)));

% Train on the Mackey glass data (only regress on middle 2000)
t = 1000:3000;
rc.train(X, T, t,10);

% Reset the reservoir to zero activations and stream the first 2000 time
% steps with "teacher forcing" (target output is used for feedback)
rc.reset();
t = 1:2000;
[~,~] = rc.stream(X(:,t),T(:,t));

% Stream the remainder without teacher forcing
t = 2001:4000;
[A,Y] = rc.stream(X(:,t));

% Plot the results, play movie of cellular automata
subplot(3,1,1);
plot(t,T(:,t),'b',t,Y,'r');
title('target output vs actual');
legend('target','actual');
xlabel('time');
ylabel('output activation');
subplot(3,1,2);
plot(t,A');
title('reservoir (plot)')
xlabel('time');
ylabel('unit activation');
subplot(3,1,3);
for t = 1:1000
    imshow(reshape(A(:,t),[dims(1),prod(dims(2:end))])/max(A(:)));
    title('reservoir over time (brightness = activation)')
    pause(1/24); % ~seconds per frame
end
