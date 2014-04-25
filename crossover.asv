function [child1, child2] = crossover(parent1, parent2)
    dim = rand < 0.5;
    dims = size(parent1);
    
    % horizontal crossover
    if (dim)
        rows = dims(1);
        row = ceil(rand * (rows-1));
        child1 = [parent1(1:row,:);parent2(row+1:rows,:)];
        child2 = [parent2(1:row,:);parent1(row+1:rows,:)];
    
    % vertical crossover
    else
        cols = dims(1);
        col = ceil(rand * (cols-1));
        child1 = [parent1(:,1:col);parent2(:,col+1:cols)];
        child2 = [parent2(:,1:col);parent1(:,col+1:cols)];
    end
end
