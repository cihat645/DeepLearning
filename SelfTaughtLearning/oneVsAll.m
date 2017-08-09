function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i 

%the 1st row of all_theta corresponds to the class for number 1

%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

X = X';

% Some useful variables
m = size(X, 1); %number of examples
n = size(X, 2); %number of features


% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1); %so all_theta matrix has dimensions: 5 x n, or 5 x 201
printf('size of all_theta\n');
size(all_theta)

% No need to Add ones to the X data matrix as the bias units have already been added 
% in the feedForwardAutoencoder.m file
X = [ones(m,1) X];
y = y';

%initial_theta = zeros(n, 1);
initial_theta = .0005 * randn(n + 1, 1);

options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels,
  printf('One vs all: ROUND %d\n',c);
  [all_theta(c,:)] = (fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options)); 
end

%the 'y == c' transforms the training label answers into 1's for every y value that is equal to  class c,
% and turns every other y-value to zero. This allows the fmincg function to train (by minimizing the cost function)
% parameters for every class, thus creating a logistic regression classifier for class c. We then store the parameters
% for the optimized classifer in the matrix "all_theta".

%for example:
%>> y = [1 ; 3; 2; 1; 5; 6; 1]
%y =
%
%   1
%   3
%   2
%   1
%   5
%   6
%   1
%
%>> (y == 1)
%ans =
%
%   1
%   0
%   0
%   1
%   0
%   0
%   1


% the matrix all_theta contains the 10 logistic regression classifiers that have been trained using the fmincg function 
% this file trains the logistic regression classifiers 











% =========================================================================


end
