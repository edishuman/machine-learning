function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    A = X * theta;                  %compute h(x)
    B = A - y;                      %compute h(x) - y
  
    %dimensional matrix to hold thetas
    theta_tmp = zeros(size(X, 2), 1);
    for iter = 1:size(X, 2)
        theta_tmp(iter) = theta(iter) - alpha/m * sum(B.*X(:, iter));
        %printf("%f - (%f * 1/%d * %f)\n", theta(iter), alpha, m, sum(arr(:, iter)));
    end

    %disp("arr2");
    %disp(arr2);

    %set thetas
    theta = theta_tmp;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
