function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n, n);
S = zeros(n, n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

Sigma = (1/m) * (X' * X); % return n x n covariance matrix

% each column of U is an eigenvector
% we are interested in the first k columns (principal components)

% S is a diagonal matrix where each diagonal entry is an eigenvalue
[U, S, V] = svd(Sigma);




% =========================================================================

end
