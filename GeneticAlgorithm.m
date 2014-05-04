classdef GeneticAlgorithm < handle
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
       % constructor for a GA
       function ga = GeneticAlgorithm(make_individual, fitness, crossover, mutate, options)
          ga.make_individual = make_individual; 
          ga.fitness = fitness;
          ga.crossover = crossover;
          ga.mutate = mutate;
          ga.options = options;
          ga.population = make_individual(); % dummy object to set class
       end
       
       % runs the GA and returns the best solution found
       %function [best, maxes, means] = evolve(ga, max_generations, population_size, num_elites, crossover_rate, mutation_rate, debug)
       function [bests, fvals] = evolve(ga, max_generations, population_size, num_elites, crossover_rate, mutation_rate, debug)
           % Prepare session to run in parallel
           if ga.options{1}
               %parpool('local', [2 10]); 
           end
           
           if nargin < 7
               debug = false;
           end
           
           ga.time = 0;
           fvals = zeros(max_generations, population_size);
           bests = cell(max_generations,1);
           %maxes = zeros(max_generations,1);
           %means = zeros(max_generations,1);
           
           % initialize population
           for i = 1:population_size
               ga.population(i) = ga.make_individual();
           end
           
           % evolve population
           while ga.time < max_generations
               ga.time = ga.time + 1;
           
               % calculate fitness
               tfit = tic;
               fit = evalFitness(ga, population_size);
               fit_time = toc(tfit);
               
               fvals(ga.time, :) = fit';
               
                % find the best individual in the population
               [~, best_idx] = max(fvals(ga.time, :));
               bests{ga.time} = ga.population(best_idx);
               
               if debug
                   disp(['Generation:  ', num2str(ga.time)]);
                   disp(['Max fitness: ', num2str(max(fvals(ga.time,:)))]);
                   disp(['Avg fitness: ', num2str(mean(fvals(ga.time,:)))]);
                   disp(['Eval time: ', num2str(fit_time)]);
                   disp(' ');
               end
               
               % get elites
               [~, idxs] = sort(fit,'descend');
               elites = ga.population(idxs(1:num_elites));
               for i = 1:num_elites
                   elites(i) = elites(i).copy();
               end
               
               % do fitness proportionate selection
               ga.population = selection(ga, fit, population_size - num_elites);
               ga.population((population_size - num_elites + 1):population_size) = elites; 
               
               % do crossover
               for i = 1:(population_size - num_elites)
                  if rand < crossover_rate(ga.time) / 2
                      r = ceil(rand * (population_size - num_elites));
                      [child1, child2] = ga.crossover(ga.population(i), ga.population(r));
                      ga.population(i) = child1;
                      ga.population(r) = child2;
                  end
               end
               
               % do mutations
               for i = 1:(population_size - num_elites)
                  ga.population(i) = ga.mutate(ga.population(i), mutation_rate(ga.time));
               end
               
           end
           
           
           % close parallel session
           if ga.options{1} 
               %delete(gcp('nocreate')); 
           end
       end
       
       % Evaluate fitness for the current population
       function fit = evalFitness(ga, population_size)
           pop = ga.population;
           if ga.options{1} % parallel
               fit = zeros(population_size, 1);
               indvFit = @(in) ga.fitness(in);
               parfor i = 1:population_size
                    fit(i) = indvFit(pop(i));
               end
           else % sequential
               popFit = @(population) arrayfun(ga.fitness, population);
               fit = popFit(pop);
           end
       end
       
       % creates a fitness proportionate list of parents
       function parents = selection(ga, fit, num_to_select)
           parents = ga.make_individual(); % dummy to set class
           
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
              %parents(i) = ga.population(count - 1);
              parents(i) = ga.population(count - 1).copy();
           end
       end
       
   end
end
