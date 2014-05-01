classdef BinaryNumber
    properties
        number;
    end
    
    methods(Static)
        function ind = create_individual(size)
            num = rand(size,1) > 0.5;
            ind = BinaryNumber(num * 1.0);
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
        
        function child = mutate(parent, mutate_chance)
            num = (rand(16,1) < (1 - mutate_chance)) == parent.number;
            child = BinaryNumber(num * 1.0);
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
        end
    end
    
    methods 
        function bn = BinaryNumber(num)
            bn.number = num;
        end
    end
end