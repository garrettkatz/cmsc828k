function child = mutate(parent, mutation_rate)
    mutations = random(size(parent)) < mutation_rate;
    signs = sign(rand(size(parent)) - 0.5);
    amount = floor(abs(normrnd(0,1,size(parent))));
    child = parent + (mutations * signs * amount);
end