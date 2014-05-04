% Cleanup variables and figures
clear;
close;

% Make single Mackey
T = makeMackeyGlass(0.5+rand,17,0.1,50000);
T = T(10001:10:end); % subsample
T = tanh(T-1); % squash into (-1,1)
X = 0.2*ones(size(T)); % constant bias

% Load speech data for individual fitness evaluation
indvFit = @(individual) individual.fitness(X, T);

% Use otcas with a 20 by 20 grid and 256 states
dims = [20 20]; % grid dimensions
K = 256;
% list of options {Parallel, ...}
% options = {true}; 
options = {false}; 

% initialize function handles for ga
crossover = @(par1,par2) OuterTotalisticCellularAutomata.crossover(par1,par2);
mutate = @(individual, rate) OuterTotalisticCellularAutomata.mutate(individual, rate);

% run ga
max_generations = 1;
population_size = 4; % Should be divisible by 4
lambda = 0.3;
for i = 1:population_size
    %initial_population(i) = OuterTotalisticCellularAutomata.smooth(dims,K,lambda); % <- good seeds
    initial_population(i) = OuterTotalisticCellularAutomata.random(dims,K,lambda);
end
num_elites = 1;
crossover_rate = @(t) 1;
mutation_rate = @(t) 0.5*(0.95^t);
gp = GeneticProgrammer(initial_population, indvFit, crossover, mutate, options);
tic
[best, fits, summaries] = gp.evolve(max_generations, num_elites, crossover_rate, mutation_rate,true);
toc

fit = indvFit(best)
plt = repmat(1:max_generations,size(fits,1),1);
scatter(plt(:),fits(:));
hold on
plot(mean(fits,1),'r+');
hold off
