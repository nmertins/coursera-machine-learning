function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1
X = [ones(m, 1) X];

z2 = X * Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2,1), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

h = a3';

% printf('Size of h: %d x %d\n', size(h));

d3 = [];
for i = 1:m
	y_vec = zeros(num_labels, 1);
	y_vec(y(i)) = 1;

	J_temp = -y_vec' * log(h(:,i)) - (1 - y_vec') * log(1 - h(:,i));
	J_temp = (1/m) * J_temp;

	J = J + J_temp;

	% grad_temp = X' * (h - y_vec');
	% grad_temp = grad_temp/m;
	% grad = grad + grad_temp;

	% printf('Size of h(:,1): %d x %d\n', size(h(:,i)));
	d3 = [d3 (h(:,i) - y_vec)];

end

theta1_reg = Theta1(:, 2:end);
theta2_reg = Theta2(:, 2:end);


d2 = theta2_reg' * d3;
d2 = d2' .* sigmoidGradient(z2);

% printf('Size of d3: %d x %d\n', size(d3));
% printf('Size of a2: %d x %d\n', size(a2));
% printf('Size of d2: %d x %d\n', size(d2));
% printf('Size of X: %d x %d\n', size(X));

Theta2_grad = d3 * a2;
Theta2_grad = (1/m) * Theta2_grad;
Theta2_grad_reg = (lambda/m) * theta2_reg;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2_grad_reg;
% printf('Size of Theta2_grad: %d x %d\n', size(Theta2_grad));
% printf('Size of Theta2_grad_reg: %d x %d\n', size(Theta2_grad_reg));

Theta1_grad = d2' * X;
Theta1_grad = (1/m) * Theta1_grad;
Theta1_grad_reg = (lambda/m) * theta1_reg;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1_grad_reg;
% printf('Size of Theta1_grad: %d x %d\n', size(Theta1_grad));
% printf('Size of Theta1_grad_reg: %d x %d\n', size(Theta1_grad_reg));

% Part 3
theta1_reg = theta1_reg.^2;
theta2_reg = theta2_reg.^2;
J = J + (lambda/(2 * m)) * (sum(theta1_reg(:)) + sum(theta2_reg(:)));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
