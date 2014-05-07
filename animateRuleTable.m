function animateRuleTable(rules, fps, figh)

if (nargin < 3)
  fig_h = gcf;
end

for i=1:length(rules)
    figure(fig_h)
    clf(fig_h)

    imshow(rules{i},[])
    pause(1/fps)
end

end
