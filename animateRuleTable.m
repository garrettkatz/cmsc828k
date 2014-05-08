%animateRuleTable(rules, fps, fig_h, doVid)
%  rules - 1-D cell array of rule tables
%  fps - frames per second
%  fig_h - figure handle to draw in (use -1 to create new figure)
%  doVid - record video, not required

function animateRuleTable(rules, fps, fig_h, doVid)

if (nargin < 4)
  doVid = false;
end
if (fig_h < 0)
  fig_h = figure;
end

ruleMax = @(i) max(max(rules{i}));
K = arrayfun(ruleMax, 1:numel(rules));
Kmax = max(K);

if doVid
  vfile = VideoWriter('rule_table.avi');
  vfile.FrameRate = 12;
  open(vfile);
end

for i=1:length(rules)
    figure(fig_h)
    clf(fig_h)

    imshow(rules{i},[0 Kmax])
    if doVid
      F = getframe;
      writeVideo(vfile, F);
    end

    pause(1/fps)
end

if doVid
  close(vfile);
end

end
