classdef BinaryNumber
    properties
        number;
    end
    
    methods(Static)
        function ind = create_individual(size)
            %num = rand(size,1) > 0.5;
            %ind = BinaryNumber(num * 1.0);
            ind = BinaryNumber(zeros(size,1));
        end
        
        function fit = fitness(ind)
            fit = dot(ind.number, [32768,16384,8192,4096,2048,1024,512,256,128,64,32,16,8,4,2,1]);
        end
        
        function [child1, child2] = crossover(parent1, parent2)
            idx = floor(rand * 15) + 1;
            num1 = [parent1.number(1:idx);parent2.number(idx+1:16)];
            num2 = [parent2.number(1:idx);parent1.number(idx+1:16)];
            
            child1 = BinaryNumber(num1);
            child2 = BinaryNumber(num2);
        end
        
        function child = mutate(parent, mutate_chance, time)
            if rand < BinaryNumber.mutation_rate(time)
                num = (rand(16,1) < (1 - mutate_chance)) == parent.number;
                child = BinaryNumber(num * 1.0);
            else 
                child = parent;
            end
        end
        
        function r = crossover_rate(time)
            r = 1;
        end
        
        function r = mutation_rate(time)
            r = 0.1;
        end
        
        function runGA()
            % Cleanup variables and figures
            clear;
            close;

            % Load speech data for individual fitness evaluation
            indvFit = @(individual) BinaryNumber.fitness(individual);

            % initialize function handles for ga
            create_individual = @() BinaryNumber.create_individual(16);
            fitness = @(population) arrayfun(indvFit, population);
            crossover = @(par1,par2) BinaryNumber.crossover(par1,par2);
            mutate = @(individual, time) BinaryNumber.mutate(individual, 0.05, time);
            crossover_rate = @(time) BinaryNumber.crossover_rate(time);
            mutation_rate = @(time) BinaryNumber.mutation_rate(time);
            options = {0};
            
            % run ga
            ga = GeneticAlgorithm(create_individual, fitness, crossover, mutate, options);
            max_generations = 100;
            population_size = 100;
            num_elites = 10;
            num_new = 10;
            debug = true;
            [best, fits] = ga.evolve(max_generations, population_size, ...
                num_elites, num_new, crossover_rate, mutation_rate, debug);

            best.number

            BinaryNumber.fitness(best)
        end
    end
    
    methods 
        function bn = BinaryNumber(num)
            bn.number = num;
        end
        
        function clone = copy(bn)
           clone = BinaryNumber(bn.number);
        end
    end
end