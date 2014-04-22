function child = crossover(parent1, parent2)
    dim = rand < 0.5;
    dims = size(parent1);
    
    % horizontal crossover
    if (dim)
        rows = dims(1);
        row = ceil(rand * (rows-1));
        child = [parent1(1:row,:);parent2(row+1:rows,:)];
    
    % vertical crossover
    else
        cols = dims(1);
        col = ceil(rand * (cols-1));
        child = [parent1(:,1:col);parent2(:,col+1:cols)];
    end
end

function child = mutate(parent, mutation_rate)
    mutations = random(size(parent)) < mutation_rate;
    signs = sign(rand(size(parent)) - 0.5);
    amount = floor(abs(normrnd(0,1,size(parent))));
    child = parent + (mutations * signs * amount);
end