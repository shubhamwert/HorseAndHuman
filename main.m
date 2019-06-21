clear; clc; close all; 
%%TO change input data or DImensions 

fprintf('\nreading horse csv file\n')

Z_horse = double(csvread('PCA_Horse_values.csv'));

fprintf('\nreading human csv file\n')

Z_human=double(csvread('PCA_Human_values.csv'));
%%assigning Y values
fprintf('\nassinging Y values...\n');

Z_horse=[Z_horse ones(size(Z_horse,1),1) zeros(size(Z_horse,1),1)];
Z_human=[Z_human zeros(size(Z_human,1),1) ones(size(Z_human,1),1) ];

fprintf('\nshuffling horse and human data....\n');

Z_horse=randomShuffle(Z_horse);

Z_human=randomShuffle(Z_human);
Z=[Z_horse;Z_human];
fprintf('\n\nsize of Z is %d \n \n',size(Z));

fprintf('\nshuffling combined data....\n');

Z=randomShuffle(Z);
Z=normalizeFeatures(Z);
fprintf('calling NN training system');
%%NN function 
[Theta1,Theta2]=TrainNN(Z);

