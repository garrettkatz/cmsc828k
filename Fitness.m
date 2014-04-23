classdef Fitness < handle
    methods(Static=true)
        function [trainErr, testErr, genErr] = evalMackey(rc, show)
        % evalMackey evaluates reservoir fitness on the Mackey-glass
        % time-series.  Errors are squashed to meliorate blow up.
        % Parameters:
        %   rc : reservoir computer object
        %   show : optional flag to plot results
        % Returns:
        %   trainErr : Mean squared error on training data seen before
        %   testErr : Mean squared error on data not seen before from the
        %       training sequence
        %   genErr: Mean squared error on data not seen before from a new
        %       sequence
            
            if nargin < 2, show = false; end;
        
            % Make time series with random initial point
            T = makeMackeyGlass(0.5+rand,17,0.1,50000);
            T = T(10001:10:end); % subsample (now have 4000 time steps)
            T = tanh(T-1); % squash into (-1,1)
            X = 0.2*ones(size(T)); % constant bias

            % Train on the Mackey glass data (only regress on second 1000)
            rc.reset();
            trainErr = rc.train(X, T, 1000:2000, 1);

            % Mean squared error on data not seen before, same series
            rc.reset();
            [~,~] = rc.stream(X(:,1:2000),T(:,1:2000));
            [~,Y] = rc.stream(X(:,2001:2500));
            testErr = (T(:,2001:2500)-Y).^2;
            testErr = mean(testErr(:));

            if show
                t = 2001:2500;
                plot(t,T(:,t)',t,Y');
            end

%             %%% Generalization error
%             % Make time series with new random initial point
%             T = makeMackeyGlass(0.5+rand,17,0.1,50000);
%             T = T(10001:10:end); % subsample (now have 4000 time steps)
%             T = tanh(T-1); % squash into (-1,1)
% 
%             % Reset the reservoir to zero activations and stream the first 2000 time
%             % steps with "teacher forcing" (target output is used for feedback)
%             rc.reset();
%             [~,~] = rc.stream(X(:,1:2000),T(:,1:2000));
% 
%             % Stream the remainder without teacher forcing
%             [~,Y] = rc.stream(X(:,2001:3000));
% 
%             % Mean squared error on data not seen before, new series
%             % genErr = tanh(T(:,2001:3000)-Y).^2;
%             genErr = (T(:,2001:3000)-Y).^2;
%             genErr = mean(genErr(:));
% 
%             if show
%                 subplot(3,1,3);
%                 t = 2001:3000;
%                 plot(t,T(:,t)',t,Y');
%             end
            genErr = 0;

        end
        function [trainErr, testErr, genErr] = evalSpeech(rc, Xsp, Tsp)
        % evalSpeech evaluates reservoir fitness on the Speech data
        % time-series.
        % Parameters:
        %   rc : reservoir computer object
        %   Xsp : input speech signals in SpeechData.mat
        %   Tsp : target output signals in SpeechData.mat
        % Returns:
        %   trainErr : 0/1 error rate on training data heard before
        %   testErr : 0/1 error rate on utterances not heard before from
        %       speakers heard before
        %   genErr : 0/1 error rate on utterances not heard before from
        %       speakers not heard before
            
            % randomize training set (N speakers, utterances, and digits)
            N = 3;
            utterances = randperm(10,N);
            speakers = randperm(10,N);
            digits = randperm(10,N);
            
            % Don't train on last speaker or utterances
            X = Xsp(utterances(1:end-1), speakers(1:end-1), digits);
            T = Tsp(utterances(1:end-1), speakers(1:end-1), digits);

            % Train
            t = repmat({1:99},size(T)); % Use 1st 99 time-steps in each series
            rc.train(X, T, t, 1);

            % Tally losses
            X = Xsp(utterances, speakers, digits);
            trainErr = 0; testErr = 0; genErr = 0;
            for u = 1:N
                for s = 1:N
                    for d = 1:N
                        % Reset the reservoir to zero activations and stream a testing signal
                        rc.reset();
                        [~,Y] = rc.stream(X{u,s,d});
                        [~,p] = max(Y(:,90)); % predicted digit
                        if s < N
                            if u < N
                                trainErr = trainErr + (p~=digits(d));
                            else
                                testErr = testErr + (p~=digits(d));
                            end
                        else
                            genErr = genErr + (p~=digits(d));
                        end
                    end
                end
            end
            trainErr = trainErr/((N-1)*(N-1)*N);
            testErr = testErr/((N-1)*N);
            genErr = genErr/(N*N);
        end
        function fit = eval(rcMackey, rcSpeech, Xsp, Tsp)
        % eval aggregates performance on both Mackey-glass and speech.
        % The two ReservoirComputers given as input should wrap the same
        % reservoir. Xsp, Tsp are the data from SpeechData.mat
            
            % Evaluate on each task
            [trainErr, testErr, genErr] = Fitness.evalMackey(rcMackey);
            [trainLoss, testLoss, genLoss] = Fitness.evalSpeech(rcSpeech, Xsp, Tsp);
            
            % Aggregate
            % unfit = trainErr+testErr+genErr+trainLoss+testLoss+genLoss;
            unfit = testErr+testLoss;
            % unfit = trainErr+trainLoss;
            %fit = 1./(1+unfit); % invert for fitness in [0,1]
            fit = 1./unfit; % invert for fitness in [0,Inf]
        end
    end
end