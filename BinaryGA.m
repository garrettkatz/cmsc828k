% Cleanup variables and figures
clear;
close;

% Load speech data for individual fitness evaluation
indvFit = @(individual) BinaryNumber.fitness(individual);

% initialize function handles for ga
create_individual = @() BinaryNumber.create_individual(16);
fitness = @(population) arrayfun(indvFit, population);
crossover = @(par1,par2) BinaryNumber.crossover(par1,par2);
mutate = @(individual) BinaryNumber.mutate(individual, 0.05);

% run ga
ga = GeneticAlgorithm(create_individual, fitness, crossover, mutate);
max_generations = 1000;
population_size = 100;
crossover_rate = 1;
mutation_rate = 0.1;
best = ga.evolve(max_generations, population_size, crossover_rate, mutation_rate);

best.number

BinaryNumber.fitness(best)