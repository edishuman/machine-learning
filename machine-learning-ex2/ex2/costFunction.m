function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%calculate J cost var
h_x = sigmoid(X * theta);
A = -y .* log(h_x);
B = (1 - y) .* log(1 - h_x);
J = sum(A - B) * 1/m;

%calculate partial deriv wrt to theta
D = h_x - y;
for iter = 1:size(theta, 1)
    grad(iter) = sum(D .* X(:, iter)) * 1/m;





% =============================================================

end
