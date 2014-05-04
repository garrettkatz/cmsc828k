% Cleanup variables and figures
clear;
close;

% Make single Mackey
T = makeMackeyGlass(0.5+rand,17,0.1,50000);
T = T(10001:10:end); % subsample
T = tanh(T-1); % squash into (-1,1)
X = 0.2*ones(size(T)); % constant bias

% Individual fitness evaluation
indvFit = @(individual) individual.fitness(X, T);

% Use a 8 x 64 grid and 4 states
dims = [8 64]; % grid dimensions
K = 4;

% list of options {Parallel, ...}
options = {false}; 

% initialize function handles for gp
lambda = 0.5;
make_individual = @() HeterogeneousCellularAutomata.random(dims,K,lambda);
crossover = @(par1,par2) HeterogeneousCellularAutomata.crossover(par1,par2);
mutate = @(individual, rate) HeterogeneousCellularAutomata.mutate(individual, rate);

% run gp
gp = GeneticProgrammer(make_individual, indvFit, crossover, mutate, options);
max_generations = 128;
population_size = 64;
num_elites = 4;
crossover_rate = @(t) 1;
mutation_rate = @(t) 0.1*(0.95^t);
tic
%best = ga.evolve(max_generations, population_size, crossover_rate, mutation_rate);
[best, fits] = gp.evolve(max_generations, population_size, num_elites, crossover_rate, mutation_rate,true);
toc

fit = indvFit(best)