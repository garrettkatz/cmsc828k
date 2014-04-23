% This file uses the reservoir computing learning technique to train an
% echo state network on the Mackey glass time series.  Since this is our
% control, I didn't spend much time optimizing the parameters.  It seems to
% work reasonably well about half of the time.  You may need to run it more
% than once to see decent results.

% Cleanup variables and figures
clear;
close;

% Generate the Mackey glass time series with a random initial point
T = makeMackeyGlass(0.5+rand,17,0.1,50000);
T = T(10001:10:end); % subsample
T = tanh(T-1); % squash into (-1,1)
X = 0.2*ones(size(T)); % constant bias

% Initialize an echo state network as in Jaeger 2010
W = 0.4*sign(sprandn(400, 400, 0.0125)); % Reservoir connections
esn = EchoStateNetwork(W);
disp('effective spectral radius:'); % Should be roughly <= 1
disp(max(abs(eigs(0.44*W + (1-0.44*0.9)*speye(400),1))));

% Wrap the echo state network in a reservoir computer object for training
% (initialize the readout matrix to all zeros)
readIn = 0.14*sign(sprandn(400,1,0.5)); % Input connections
readOut = zeros(1,400);
readBack = 0.56*(2*rand(400,1)-1); % Feedback connections
rc = ReservoirComputer(esn, readIn, readOut, readBack);

% Train on the Mackey glass data (only regress on middle 2000)
rc.train(X, T, 1000:3000);

% Reset the reservoir to zero activations and stream the first 3000 time
% steps with "teacher forcing" (target output is used for feedback)
rc.reset();
[~,~] = rc.stream(X(:,1:3000),T(:,1:3000));

% Stream the remainder without teacher forcing
t = 3001:4000;
[A,Y] = rc.stream(X(:,t));

% Plot the results
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
imshow(A);
title('reservoir (brightness = activation)')
xlabel('time');
ylabel('unit index');
