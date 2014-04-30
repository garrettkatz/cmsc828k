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
       function best = evolve(ga, max_generations, population_size, crossover_rate, mutation_rate)
           % Prepare session to run in parallel
           if ga.options{1}
               parpool('local', 2); 
           end
           
           ga.time = 0;
           
           % initialize population
           for i = 1:population_size
               ga.population(i) = ga.make_individual();
           end
           
           % evolve population
           while ga.time < max_generations
               ga.time = ga.time + 1;
           
               % do fitness proportionate selection
               ga.population = selection(ga, population_size);
               
               % do crossover
               for i = 1:population_size
                  if rand < crossover_rate / 2
                      r = ceil(rand * population_size);
                      [child1, child2] = ga.crossover(ga.population(i), ga.population(r));
                      ga.population(i) = child1;
                      ga.population(r) = child2;
                  end
               end
               
               % do mutations
               for i = 1:population_size
                  if rand < mutation_rate
                      ga.population(i) = ga.mutate(ga.population(i));
                  end
               end
           end
           
           % find the best individual in the population
           [~, best_idx] = max(evalFitness(ga, population_size));
           best = ga.population(best_idx);
           
           % close parallel session
           if ga.options{1} 
               delete(gcp('nocreate')); 
           end
       end
       
       % Evaluate fitness for the current population
       function fit = evalFitness(ga, population_size)
           pop = ga.population;
           if ga.options{1} % parallel
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
       function parents = selection(ga, num_to_select)
           parents = ga.make_individual(); % dummy to set class
           
           % calculate fitness
           fit = evalFitness(ga, num_to_select);
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
              parents(i) = ga.population(count - 1);
           end
       end
       
   end
end