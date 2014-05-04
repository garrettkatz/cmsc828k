classdef GeneticProgrammer < handle
   properties
       population;
       make_individual;
       time;
       fitness;
       crossover;
       mutate;
       options;
   end
   
   methods
       % constructor for a GP
       function gp = GeneticProgrammer(initial_population, fitness, crossover, mutate, options)
          gp.fitness = fitness;
          gp.crossover = crossover;
          gp.mutate = mutate;
          gp.options = options;
          P = numel(initial_population);
          gp.population = initial_population(1:P-mod(P,4));
       end
       
       % runs the GP and returns the best solution found
       function [best, fits, summaries] = evolve(gp, max_generations, num_elites, crossover_rate, mutation_rate, debug)
           % Prepare session to run in parallel
           if gp.options{1}
               parpool('local', 2); 
           end
           
           if nargin < 6
               debug = false;
           end
           
           population_size = numel(gp.population);
           
           gp.time = 0;
           fits = zeros(population_size, max_generations);
           summaries = cell(population_size, max_generations);
           
           % evolve population
           while gp.time < max_generations
               gp.time = gp.time + 1;
           
               % calculate fitness
               fit = evalFitness(gp, population_size);
               fits(:,gp.time) = fit;
               for i = 1:population_size
                   summaries{i,gp.time} = gp.population(i).summary();
               end
               
               if debug
                   disp(['Generation:  ', num2str(gp.time)]);
                   disp(['Max fitness: ', num2str(max(fit(:)))]);
                   disp(['Avg fitness: ', num2str(mean(fit(:)))]);
                   disp(' ');
               end
               
               % choose survivors (better half)
               [~, idxs] = sort(fit,'descend');
               survivors = gp.population(idxs(1:population_size/2));
               
               % save elites
               elites = survivors(1:num_elites);
               
               % permute randomly for pairing
               survivors = survivors(randperm(population_size/2));
               
               % do two crossovers per pair
               for i = 1:population_size/4
                   [gp.population(4*i-3), gp.population(4*i-2)] = ...
                       gp.crossover(survivors(2*i-1), survivors(2*i));
                   [gp.population(4*i-1), gp.population(4*i)] = ...
                       gp.crossover(survivors(2*i-1), survivors(2*i));
               end
               
               % do mutations
               for i = 1:population_size
                  gp.population(i) = gp.mutate(gp.population(i), mutation_rate(gp.time));
               end
               
               % restore elites (last in pop are roughly least fit)
               gp.population(end-num_elites+1:end) = elites;
               
           end
           
           % find the best individual in the population
           [~, best_idx] = max(evalFitness(gp, population_size));
           best = gp.population(best_idx);
           
           % close parallel session
           if gp.options{1} 
               delete(gcp('nocreate')); 
           end
       end
       
       % Evaluate fitness for the current population
       function fit = evalFitness(gp, population_size)
           pop = gp.population;
           if gp.options{1} % parallel
               indvFit = @(in) gp.fitness(in);
               parfor i = 1:population_size
                    fit(i) = indvFit(pop(i));
               end
           else % sequential
               popFit = @(population) arrayfun(gp.fitness, population);
               fit = popFit(pop);
           end
       end
       
       % creates a fitness proportionate list of parents
       function parents = selection(gp, fit, num_to_select)
           parents = gp.make_individual(); % dummy to set class
           
           % calculate fitnes
           fit = fit / sum(fit);
           
           % for each survivor
           for i = 1:num_to_select
              
              % pick with replacement weighted by fitness 
              count = 1;
              r = rand;
              while (r > 0)
                  r = r - fit(count);
                  count = count + 1;
              end
              
              % throw in survivors vector
              parents(i) = gp.population(count - 1);
              %parents(i) = ga.population(count - 1).copy();
           end
       end
       
   end
end