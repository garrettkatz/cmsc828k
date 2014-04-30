% Cleanup variables and figures
clear;
close;

% Load speech data for individual fitness evaluation
load SpeechData.mat % loads inputs Xsp and targets Tsp
indvFit = @(individual) Fitness.evalOtca(individual, Xsp, Tsp);

% Use otcas with a 20 by 20 grid and 256 states
dims = [20 20]; % grid dimensions
K = 256;
% list of options {Parallel, ...}
options = {true}; 

% initialize function handles for ga
make_individual = @() OuterTotalisticCellularAutomata.random(dims,K);
crossover = @(par1,par2) OuterTotalisticCellularAutomata.crossover(par1,par2);
mutate = @(individual) OuterTotalisticCellularAutomata.mutate(individual, 0.1);

% run ga
ga = GeneticAlgorithm(make_individual, indvFit, crossover, mutate, options);
max_generations = 2;
population_size = 2;
crossover_rate = 1;
mutation_rate = 0.1;
tic
best = ga.evolve(max_generations, population_size, crossover_rate, mutation_rate);
toc

% evaluate best's performance on mackey-glass
N = numel(best.a);
ext = randperm(N, 20); % indices of external-signal-receiving units
readIn = sparse(ext(1:10), 1, 1, N, 1); % 1st 10 for input
readOut = zeros(1, N);
readBack = sparse(ext(11:20), 1, 1, N, 1); % last 10 for feedback
rcMackey = ReservoirComputer(best, readIn, readOut, readBack);
[trainErr, testErr, ~] = Fitness.evalMackey(rcMackey, true)
