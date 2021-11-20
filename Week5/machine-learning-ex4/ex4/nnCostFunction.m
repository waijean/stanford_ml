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
a_1 = [ones(m, 1), X]; % returns m x 401 matrix

% hidden layer
z_2 = a_1 * Theta1'; % returns m x 25 matrix
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1), a_2];

% output layer
z_3 = a_2 * Theta2'; % returns m x 10 matrix
a_3 = sigmoid(z_3);

% expand y into matrix
y_matrix = eye(num_labels)(y,:); % returns m x 10 matrix

% J should be a scalar
J = (1/m) *sum(sum((-y_matrix .* log(a_3)) - ((1-y_matrix) .* log(1-a_3))));

% exclude bias terms in reg term
reg_theta = sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2));

% J with reg term
J = J + (lambda/(2*m) * reg_theta);

% Part 2
for t = 1:m
  % Step 1: Perform feedforward pass
  a_1 = [1; X(t,:)']; % returns 401 x 1 column vector
  % hidden layer
  z_2 = Theta1 * a_1; % returns 25 x 1 column vector
  a_2 = sigmoid(z_2); 
  a_2 = [1; a_2]; % returns 26 x 1 column vector
  % output layer
  z_3 = Theta2 * a_2; % returns 10 x 1 column vector
  a_3 = sigmoid(z_3);
  
  % Step 2: Compute delta_3
  y_vector = eye(num_labels)(y(t),:)'; % returns 10 x 1 column vector
  delta_3 = a_3 - y_vector;

  % Step 3: Compute delta_2
  delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z_2)]; % returns 26 x 1 column vector
  delta_2 = delta_2(2:end); % removing the bias term
  
  % Step 4: Accumulate the gradient
  Theta2_grad = Theta2_grad + (delta_3 * a_2'); % returns 10 x 26 matrix
  Theta1_grad = Theta1_grad + (delta_2 * a_1'); % returns 25 x 401 matrix
  
endfor

% unregularized gradient
Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;

% exclude bias column in reg term
Theta2_grad_reg = (lambda/m) * Theta2;
Theta2_grad_reg(:,1) = 0;

Theta1_grad_reg = (lambda/m) * Theta1;
Theta1_grad_reg(:,1) = 0;

% regularized gradient
Theta2_grad = Theta2_grad + Theta2_grad_reg;
Theta1_grad = Theta1_grad + Theta1_grad_reg;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
