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
       function [bests, fits] = evolve(ga, max_generations, population_size, ...
               num_elites, num_new, crossover_rate, mutation_rate, debug)
           % Prepare session to run in parallel
           if ga.options{1}
               %parpool('local', [2 10]); 
           end
           
           if nargin < 7
               debug = false;
           end
           
           ga.time = 0;
           fits = zeros(population_size, max_generations);
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
          
               fits(:,ga.time) = fit;
               
                % find the best individual in the population
               [~, best_idx] = max(fits(:,ga.time));
               bests{ga.time} = ga.population(best_idx);
               
               if debug
                   disp(['Generation:  ', num2str(ga.time)]);
                   disp(['Max fitness: ', num2str(max(fit))]);
                   disp(['Avg fitness: ', num2str(mean(fit))]);
                   disp(['Eval time: ', num2str(fit_time)]);
                   disp(' ');
               end
               
               % get elites
               [~, idxs] = sort(fit,'descend');
               elites = ga.population(idxs(1:num_elites));
               for i = 1:num_elites
                   elites(i) = elites(i).copy();
               end
               
               num_to_select = population_size - (num_elites + num_new);
               
               % do fitness proportionate selection
               ga.population = top_half_selection(ga, fit, num_to_select);
               
               % add new members
               for i = 1:num_new
                   ga.population(num_to_select + i) = ga.make_individual(); 
               end
               ga.population = ga.population(randperm(population_size - num_elites));
               
               % add elites
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
       
       % selects the top half of the population
       function parents = top_half_selection(ga, fit, num_to_select)
           [~, idxs] = sort(fit,'descend');
           half = floor(num_to_select / 2);
           parents = [ga.population(idxs(1:half));ga.population(idxs(1:half))];
           if mod(num_to_select, 2)
              parents(num_to_select) = ga.population(half + 1); 
           end
           
           parents = parents(randperm(num_to_select));
       end
       
       % creates a fitness proportionate list of parents
       function parents = fitness_prop_selection(ga, fit, num_to_select)
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
