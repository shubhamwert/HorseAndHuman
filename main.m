%%TO change input data or DImensions 
[Z_horse,Z_human] = [csvread('PCA_Horse_values'),csvread('PCA_Human_values')]

%%assigning Y values

Z_horse=[Z_horse ones(size(Z_horse,1),1) zeros(size(Z_horse,1),1)];
Z_horse=randomShuffle(Z_horse);
Z_human=[Z_human zeros(size(Z_human,1),1) ones(size(Z_human,1),1) ];
Z_human=randomShuffle(Z_human);
Z=[Z_horse;Z_human];
fprintf('\n\nsize of Z is %d \n \n',size(Z));

Z=randomShuffle(Z);

%%NN function 
[Theta1,Theta2]=TrainNN(Z);

