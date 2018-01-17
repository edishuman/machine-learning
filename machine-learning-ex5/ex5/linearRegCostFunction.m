function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%cost function
reg = lambda / (2*m) * sum(theta(2:end) .^ 2);
A = X * theta;
B = A - y;
C = B .^ 2;
J = sum(C) / (2*m) + reg;

%gradient, needs to work for any amount of theta, no hardcode
for iter = 2:size(theta, 1)
    grad(iter) = 1/m * sum(B .* X(:, iter)) + (lambda/m * theta(iter));
grad(1) = 1/m * sum(B .* X(:, 1));





% =========================================================================

grad = grad(:);

end
