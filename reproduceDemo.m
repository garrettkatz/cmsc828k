%Call this first, with your desired parameters
%Then call animateRuleTable(rules(i,:), frames_per_second)

clear
clear functions

Nidv = 3;
Ngen = 100;
dims = [20 20];
K = 100;

rate = @(t) 0.2*exp(-4*t/Ngen);
make_individual = @() OuterTotalisticCellularAutomata.gauss(dims,K,0.8);
crossover = @(par1,par2) OuterTotalisticCellularAutomata.smoothCrossover(par1,par2);
mutate = @(individual, rate) OuterTotalisticCellularAutomata.gaussMutate(individual, rate, false);

rules = cell(Nidv, Ngen);
for i=1:Nidv
  pop_old(i) = make_individual();
  pop_new(i) = make_individual();
end

for t=1:Ngen
  disp(['Generation ',num2str(t)])
  for i=1:Nidv
    rules{i,t} = pop_old(i).rule;
    %random crossover
    cross_opts = setdiff(1:Nidv,i);
    cross_idx = cross_opts(randi(length(cross_opts)));

    [child1, ~] = crossover(pop_old(i), pop_old(cross_idx));
    pop_new(i) = mutate(child1, rate(t));

  end
  pop_old = pop_new;
end
