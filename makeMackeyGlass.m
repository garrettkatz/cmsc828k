function Y = makeMackeyGlass(y1, tau, delta, length)
%MAKEMACKEYGLASS Generates discrete-time approximation of Mackey-Glass
    % Fixed parameters
    alpha = 0.2;
    beta = 10;
    gamma = 0.1;
    % offset between current and delayed time step
    offset = round(tau/delta);
    % Initialize output
    Y = [y1 zeros(1,length-1)];
    % Generate sequence
    for t = 1:length-1
        if t-offset < 1
            Y(t+1) = Y(t) - delta*gamma*Y(t); % Assume Y(t-offset) = 0
        else
            Y(t+1) = Y(t) + ...
                delta*(alpha*Y(t-offset)/(1+Y(t-offset)^beta)-gamma*Y(t));
        end
    end
end

