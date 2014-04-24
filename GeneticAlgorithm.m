classdef GeneticAlgorithm < handle
   properties
       population;
       make_individual;
       time;
       fitness;
       crossover;
       mutate;
   end
   
   methods
       % constructor for a GA
       function ga = GeneticAlgorithm(make_individual, fitness, crossover, mutate)
          ga.make_individual = make_individual; 
          ga.fitness = fitness;
          ga.crossover = crossover;
          ga.mutate = mutate;
          ga.population = make_individual(); % dummy object to set class
       end
       
       % runs the GA and returns the best solution found
       function best = evolve(ga, max_generations, population_size, crossover_rate, mutation_rate)
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
           [~, best_idx] = max(ga.fitness(ga.population));
           best = ga.population(best_idx);
       end
       
       % creates a fitness proportionate list of parents
       function parents = selection(ga, num_to_select)
           parents = ga.make_individual(); % dummy to set class
           
           % calculate fitness
           fit = ga.fitness(ga.population);
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