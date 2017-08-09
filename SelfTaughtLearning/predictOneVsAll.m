function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1); %remember, size(X,1) returns the number of rows in the matrix
num_labels = size(all_theta, 1) %num_labels represents the number of classes or categories there are, in this problem it's 10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

prediction = all_theta * X'; %so prediction has dimensions 10 x 5000 
prediction = prediction';  %changes dimensions to 5000 x 10
printf('sampling prediction values...\n');
prediction(1:10,:)

[highest_value, prediction] = max(prediction, [], 2); %this returns the maximum value for each rows
%the values in each row of prediction correspond to the likelihood that the class (represented by each column) fits the data for that row


p = prediction;




%the problem I had at first is that the prediction vector had the maximum values for each class, but we haven't converted those maximum values to the corresponding class values
% for instance, say the prediction value is 4.23 and it's the max value
% 4.23 isn't a possible number in this situation, so for whichever column that was in we need to convert the 4.23 to a 3 
% if the value 4.23 was taken from the third column 

%the solution to this problem is to store the index of the max value in the prediction vectorize

% the max function notation is as follows

% [max_value, max_value's_index] = max(data);








% =========================================================================


end
