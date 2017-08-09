function [all_theta] = oneVsAllOG(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i 

%the 1st row of all_theta corresponds to the class for number 1

%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1); %so all_theta matrix has dimensions: 10 x n + 1 or, 10 x 401

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

initial_theta = zeros(n + 1, 1);

options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels,
   printf('ROUND %d\n',c);
  [all_theta(c,:)] = (fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options)); 
end

%the 'y == c' transforms the training label answers into 1's for every y value that is equal to  class c,
% and turns every other y-value to zero. This allows the fmincg function to train (by minimizing the cost function)
% parameters for every class, thus creating a logistic regression classifier for class c. We then store the parameters
% for the optimized classifer in the matrix "all_theta".

% the matrix all_theta contains the 10 logistic regression classifiers that have been trained using the fmincg function 
% this file trains the logistic regression classifiers 



% =========================================================================

end
