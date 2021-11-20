function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
h = sigmoid(X * theta); % m x 1 column vector

error = (1/m) * sum(-y .* log(h) - (1-y) .* log(1-h));
reg_term = (lambda / (2*m)) * sum(theta(2:end) .^ 2);

% the cost function is the sum of error term and regularization term
J = error + reg_term;

grad = (1/m) .* (X' * (h-y)); 

% add the regularization gradient except for theta(1) 
grad(2:end) = grad(2:end) + (lambda / m) .* theta(2:end);



% =============================================================

end
