% Cleanup variables and figures
clear;
close;

% Make single Mackey
T = makeMackeyGlass(0.5+rand,17,0.1,50000);
T = T(10001:10:end); % subsample
T = tanh(T-1); % squash into (-1,1)
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
mutate = @(individual, rate) OuterTotalisticCellularAutomata.gaussMutate(individual, rate);

% run ga
ga = GeneticAlgorithm(make_individual, indvFit, crossover, mutate, options);
max_generations = 5;
population_size = 5;
num_elites = 1;
num_new = 1;
crossover_rate = @(t) 0.8;
mutation_rate = @(t) 0.1*exp(-t/(0.5*max_generations));
tic

disp('Starting evolution...')
[bests, fits] = ga.evolve(max_generations, population_size, num_elites, num_new, crossover_rate, mutation_rate,true);
toc

% evaluate best's performance on mackey-glass
fit = indvFit(best)
best.check(X,T,[20 20],100,1/48)
plt = repmat(1:max_generations,size(fits,1),1);
scatter(plt(:),fits(:));
hold on
plot(mean(fits,1),'r+');
hold off
