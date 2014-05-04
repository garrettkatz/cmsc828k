function animateRuleTable(rules,fps)

for i=1:length(rules)
    imshow(rules{i},[])
    pause(1/fps)
end

end