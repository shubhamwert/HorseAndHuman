function [Theta1,Theta2] = TrainNN(data)

%%just like that to be sure of data being random
data=randomShuffle(data);
X_main=data(:,1:end-2);
Y_main=data(:,end-1:end);
size(Y_main)
pause;

%%sanity check

%%seprating training set

X=X_main(1:850,:);
y=Y_main(1:850,:);

x_cv=X_main(800:950,:);
y_cv=Y_main(800:950,:);

x_test=X_main(950:end,:);
y_test=Y_main(950:end,:);


%%NN parameters
fprintf('\nInitializing NN params... \n')

input_layer_size  = size(X,2); 
hidden_layer_size = 2500; 
labels = 2;  
                          

initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)/10;
initial_Theta2 = randInitializeWeights(hidden_layer_size,labels)/10;

size(initial_Theta1)
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

lambda=0.1;
%%testing J


J_t= nnCostFunction(initial_nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   labels, X, y, lambda)


fprintf('\n\nTraining Neural Network... \n\n')

options = optimset('MaxIter', 5);

lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   labels, X, y, lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%%CV
fprintf("performing cross validation...\n");
lambda_history=zeros(1000,1);
fprintf('iterating lambda.')
for i=1:1000
    fprintf('..')
    lambda_history(i)= nnCostFunction(nn_params,input_layer_size,hidden_layer_size,labels,x_cv, y_cv, i/10);
end
[minval,row]=min(min(lambda_history));
lambda=row;
fprintf('\n\n')
error_test=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,labels,X_test, Y_test, lambda)*100/length(X_test)

error_on_full_data=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,labels,X_main, Y_main, lambda)*100/length(X_test)

end