% Cleanup variables and figures
clear;
close;

% Load speech data for individual fitness evaluation
indvFit = @(individual) BinaryNumber.fitness(individual);

% initialize function handles for ga
create_individual = @() BinaryNumber.create_individual(16);
fitness = @(population) arrayfun(indvFit, population);
crossover = @(par1,par2) BinaryNumber.crossover(par1,par2);
mutate = @(individual, rate) BinaryNumber.mutate(individual, rate);
options = {false}; 

% run ga
ga = GeneticAlgorithm(create_individual, fitness, crossover, mutate, options);
max_generations = 1000;
population_size = 100;
crossover_rate = @(t) 1;
mutation_rate = @(t) 0.05;
num_elites = 1;
[best, maxes, means] = ga.evolve(max_generations, population_size, num_elites, crossover_rate, mutation_rate);

best.number

BinaryNumber.fitness(best)