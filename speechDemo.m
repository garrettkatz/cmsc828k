% This file uses the reservoir computing learning technique to train an
% echo state network on the speech data.

% Cleanup variables and figures
clear;
close;

% Initialize an echo state network as in Jaeger 2010
W = 0.4*sign(sprandn(400, 400, 0.0125)); % Reservoir connections
esn = EchoStateNetwork(W);
disp('effective spectral radius:'); % Should be roughly <= 1
disp(max(abs(eigs(0.44*W + (1-0.44*0.9)*speye(400),1))));

% Wrap the echo state network in a reservoir computer object for training
readIn = 0.14*sign(sprandn(400,13,0.5)); % Input connections (13 channels)
readOut = zeros(10,400);
readBack = zeros(400,10); % No feedback (10 channels)
rc = ReservoirComputer(esn, readIn, readOut, readBack);

% Load data and randomize training set
load('SpeechData.mat');
answers = repmat(reshape(1:10,1,1,[]), [10 10 1]); % correct answers
N = 3;
utterances = randperm(10,N);
speakers = randperm(10,N);
digits = randperm(10,N);
% Leave last speaker and utterances for testing
X = Xsp(utterances(1:end-1), speakers(1:end-1), digits);
T = Tsp(utterances(1:end-1), speakers(1:end-1), digits);

% Train
t = repmat({1:99},size(T)); % Use 1st 99 time-steps in each series
rc.train(X, T, t);

% Reset the reservoir to zero activations and stream a testing signal
rc.reset();
u = utterances(1); s = speakers(1); d = digits(1); % nothing new
% u = utterances(end); s = speakers(1); d = digits(1); % new utterance
% u = utterances(end); s = speakers(end); d = digits(1); % new speaker
[A,Y] = rc.stream(Xsp{u,s,d});
disp('correct answer:');
disp(answers(u,s,d));
disp('actual answer:');
[~,a] = max(Y(:,90));
disp(a);
disp('output channels:');
disp(Y(:,90)');
% Plot the results
t = 1:100;
subplot(3,1,1);
plot(t,Xsp{u,s,d}');
title('input signal (13 channels)')
xlabel('time');
ylabel('MFCC');
subplot(3,1,2);
plot(t,A');
title('reservoir (plot)')
xlabel('time');
ylabel('unit activation');
subplot(3,1,3);
plot(t,Tsp{u,s,d}','b',t,Y','r');
title('target output vs actual');
xlabel('time');
ylabel('output activation');
