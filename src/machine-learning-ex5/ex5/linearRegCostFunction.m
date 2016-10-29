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


% printf('Size of X: %d x %d\n', size(X));
% printf('Size of y: %d x %d\n', size(y));
% printf('Size of theta: %d x %d\n', size(theta));


h = X * theta;
err = h - y;
sqr_err = err' * err;
sqr_err = (1/(2*m)) * sqr_err;


theta_reg = theta(2:end,:);
reg_term = theta_reg' * theta_reg;
reg_term = (lambda/(2*m)) * reg_term;


J = sqr_err + reg_term;

% printf('Size of err_grad: %d x 1\n', size(theta, 1))
grad = (1/m) * (X' * err);
% printf('Size of grad: %d x %d\n', size(grad));
grad(2:end,:) = grad(2:end,:) + ((lambda/m) * theta(2:end,:));


% =========================================================================

grad = grad(:);

end
