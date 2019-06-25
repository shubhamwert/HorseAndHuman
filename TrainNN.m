function [Theta1,Theta2,mu,sigma] = TrainNN(data)



fprintf('\n\n');
%%just like that to be sure of data being random
data=randomShuffle(data);


X_main=data(:,1:end-2);
Y_main=data(:,end-1:end);

%%sanity check
%[X_main,mu,sigma]=normalizeFeatures(X_main);
%%seprating training set

X=X_main(1:850,:);
y=Y_main(1:850,:);
x_cv=X_main(800:950,:);
y_cv=Y_main(800:950,:);
x_test=X_main(950:end,:);
y_test=Y_main(950:end,:);


%%NN parameters
fprintf('\nInitializing NN params... \n')
labels = 2;  
input_layer_size  = size(X,2); 
hidden_layer_size = 2*floor(sqrt(input_layer_size*labels));   %masters 

                          

initial_Theta1 = randInitializeWeights(hidden_layer_size, input_layer_size);
initial_Theta2 = randInitializeWeights(labels, hidden_layer_size);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

lambda=0.01;
%%testing J


J_t= nnCostFunction(initial_nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   labels, X, y, lambda)


fprintf('\n\nTraining Neural Network... \n\n')

options = optimset('MaxIter', 9000);

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
lambda_history=zeros(100,1);
fprintf('iterating lambda.')
for i=1:100
    fprintf('.')
    lambda_history(i)= nnCostFunction(nn_params,input_layer_size,hidden_layer_size,labels,x_cv, y_cv, i/10);
end

plot(0.1:0.1:10,lambda_history,'x');

[minval,row]=min(min(lambda_history));
lambda=row;
fprintf('calculating theta with regulariztion with choosen lambda = %d',lambda);
costFunction = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   labels, X, y, lambda);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 labels, (hidden_layer_size + 1));


fprintf('\n\n')
error_test=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,labels,x_test, y_test, lambda)*100

error_on_full_data=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,labels,X_main, Y_main, lambda)*100/length(X_main)


pred = predict(Theta1, Theta2,x_test);


fprintf('\n test Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

fprintf('Program paused. Press enter to continue.\n');
fprintf('if accuracy is 0 re-run the function as theta might not be correctly initialize \n');

pause;
m=sum(sqrt((pred-y_test).^2))
fprintf('error case through y == ',m)


end