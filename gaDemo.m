% Cleanup variables and figures
clear;
close;

% Make single Mackey
T = makeMackeyGlass(0.5+rand,17,0.1,50000);
T = tanh(T-1); % squash into (-1,1)

%Make sine wave time series
%T = makeSineSeries(3,[1 6],50,1000);

T = T(10001:10:end); % subsample
X = 0.2*ones(size(T)); % constant bias

% Load speech data for individual fitness evaluation
%load SpeechData.mat % loads inputs Xsp and targets Tsp
%indvFit = @(individual) Fitness.evalOtca(individual, Xsp, Tsp);
indvFit = @(individual) individual.fitness(X, T);

% Use otcas with a 20 by 20 grid and 256 states
dims = [20 20]; % grid dimensions
K = 256;
% list of options {Parallel, ...}
% options = {true}; 
options = {true}; 

% initialize function handles for ga
%make_individual = @() OuterTotalisticCellularAutomata.random(dims,K,0.3);
%make_individual = @() OuterTotalisticCellularAutomata.smooth(dims,K,0.3);
make_individual = @() OuterTotalisticCellularAutomata.gauss(dims,K);
crossover = @(par1,par2) OuterTotalisticCellularAutomata.smoothCrossover(par1,par2);
%mutate = @(individual, rate) OuterTotalisticCellularAutomata.mutate(individual, rate);
mutate = @(individual, rate) OuterTotalisticCellularAutomata.gaussMutate(individual, rate, true); % true for cts mutation

% run ga
ga = GeneticAlgorithm(make_individual, indvFit, crossover, mutate, options);
max_generations = 5;%500;
population_size = 5;%100;
num_elites = 1;%5;
num_new = 1;%5;
crossover_rate = @(t) 0.8;
mutation_rate = @(t) 0.5*(0.975^t);
tic

disp('Starting evolution...')
[bests, fits, summaries] = ga.evolve(max_generations, population_size, num_elites, num_new, crossover_rate, mutation_rate,true);
toc
best = bests{end};

% Compare fitness against time
subplot(1,2,1);
plt = repmat(1:max_generations,size(fits,1),1);
scatter(plt(:),fits(:));
hold on
plot(mean(fits,1),'r-+');
hold off

% compare lambdas/cts against fitness
lambdas = summaries(:,:,1);
wlambdas = summaries(:,:,2);
ctss = summaries(:,:,3);
subplot(1,2,2);
plot(fits(:),lambdas(:),'ro',fits(:),wlambdas(:),'gd',fits(:),ctss(:),'b+');
legend('lambda','wlambda','cts');

% evaluate best's performance on mackey-glass
fit = indvFit(best);
figure()
best.check(X,T,[20 20],100,1/48);
