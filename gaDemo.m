% Cleanup variables and figures
clear;
close;

% Make single Mackey
T = makeMackeyGlass(0.5+rand,17,0.1,50000);
T = T(10001:10:end); % subsample
T = tanh(T-1); % squash into (-1,1)
X = 0.2*ones(size(T)); % constant bias

% Load speech data for individual fitness evaluation
load SpeechData.mat % loads inputs Xsp and targets Tsp
%indvFit = @(individual) Fitness.evalOtca(individual, Xsp, Tsp);
indvFit = @(individual) individual.fitness(X, T, 1000:3000);

% Use otcas with a 20 by 20 grid and 256 states
dims = [20 20]; % grid dimensions
K = 256;
% list of options {Parallel, ...}
% options = {true}; 
options = {false}; 

% initialize function handles for ga
make_individual = @() OuterTotalisticCellularAutomata.random(dims,K);
crossover = @(par1,par2) OuterTotalisticCellularAutomata.crossover(par1,par2);
mutate = @(individual) OuterTotalisticCellularAutomata.mutate(individual, 0.1);

% run ga
ga = GeneticAlgorithm(make_individual, indvFit, crossover, mutate, options);
max_generations = 5;
population_size = 5;
num_elites = 1;
crossover_rate = @(i) 1;
mutation_rate = @(i) 0.2;
tic
%best = ga.evolve(max_generations, population_size, crossover_rate, mutation_rate);
[best, maxes, means] = ga.evolve(max_generations, population_size, num_elites, crossover_rate, mutation_rate,true);
toc

% evaluate best's performance on mackey-glass
% N = numel(best.a);
% ext = randperm(N, 20); % indices of external-signal-receiving units
% readIn = sparse(ext(1:10), 1, 1, N, 1); % 1st 10 for input
% readOut = zeros(1, N);
% readBack = sparse(ext(11:20), 1, 1, N, 1); % last 10 for feedback
% rcMackey = ReservoirComputer(best, readIn, readOut, readBack);
% [trainErr, testErr, ~] = Fitness.evalMackey(rcMackey, true)
fit = indvFit(best)