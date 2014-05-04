% Cleanup variables and figures
clear;
close;

% Load speech data for individual fitness evaluation
indvFit = @(individual) BinaryNumber.fitness(individual);

% initialize function handles for ga
fitness = @(population) arrayfun(indvFit, population);
crossover = @(par1,par2) BinaryNumber.crossover(par1,par2);
mutate = @(individual, rate) BinaryNumber.mutate(individual, rate);
options = {false}; 

% run gp
max_generations = 100;
population_size = 100; % should be divisible by 4
for i = 1:population_size
    initial_population(i) = BinaryNumber.create_individual(16);
end
crossover_rate = @(t) 1;
mutation_rate = @(t) 0.1*(0.95^t);
num_elites = 1;
gp = GeneticProgrammer(initial_population, fitness, crossover, mutate, options);
[best, fits] = gp.evolve(max_generations, num_elites, crossover_rate, mutation_rate, true);

%best.number

BinaryNumber.fitness(best)
plt = repmat(1:max_generations,size(fits,1),1);
scatter(plt(:),fits(:));
hold on
plot(mean(fits,1),'r+');
hold off
