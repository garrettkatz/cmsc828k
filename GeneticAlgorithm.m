classdef GeneticAlgorithm
   properties
       population;
       population_creator;
       time;
       fitness;
       crossover;
       mutation;
   end
   
   methods
       % runs the GA and returns the best solution found
       function best = evolve(ga, max_generations, population_size, crossover_rate, mutation_rate)
           ga.time = 0;
           ga.population = ga.population_creator(population_size);
           
           while ga.time < max_generations
               ga.time = ga.time + 1;
              
               ga.population = selection(ga, population_size);
               
           end
       end
       
       % creates a fitness proportionate list of parents
       function parents = selection(ga, num_to_select)
           parents = zeros(1,num_to_select);
           
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