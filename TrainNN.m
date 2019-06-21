function [Theta1,Theta2] = TrainNN(data)

%%just like that to be sure of data being random
data=randomShuffle(data);

X_main=data(1:end-2,:);
Y_main=data(end-1:end,:);


%%sanity check
fprintf('Size of Y_main is %d',size(Y_main));

%%seprating training set

X=X_main(1:850,:);
y=Y_main(1:850,:);

x_cv=X_main(800:950,:);
y_cv=Y_main(800:950,:);

x_train=X_main(950:end,:);
y_train=Y_main(950:end,:);




end