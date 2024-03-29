disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data',name,'_train.csv'));
X = data(:,1:2);
Y = data(:,3);

% Carry out training, primal and/or dual
%%% TODO %%%

% Define the predictSVM(x) function, which uses trained parameters
%%% TODO %%%


hold on;
% plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], 'SVM Train');


disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

% plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], 'SVM Validate');

