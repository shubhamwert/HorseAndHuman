clear; clc; close all; 
%%TO change input data or DImensions 

fprintf('\nreading horse csv file\n')

Z_horse_main = double(csvread('PCA_Horse_values.csv'));

fprintf('\nreading human csv file\n')

Z_human_main=double(csvread('PCA_Human_values.csv'));

%%assigning Y values
fprintf('\nassinging Y values...\n');

Z_horse=[Z_horse_main ones(size(Z_horse_main,1),1) zeros(size(Z_horse_main,1),1)];
Z_human=[Z_human_main zeros(size(Z_human_main,1),1) ones(size(Z_human_main,1),1) ];

fprintf('\nshuffling horse and human data....\n');

Z_horse=randomShuffle(Z_horse);
Z_human=randomShuffle(Z_human);

Z=[Z_horse;Z_human];
fprintf('\n\nsize of Z is %d \n \n',size(Z));

fprintf('\nshuffling combined data....\n');

Z=randomShuffle(Z);

fprintf('\ncalling NN training system\n');
%%NN function 
[Theta1,Theta2]=TrainNN(Z);

fprintf('\ntesting real time\n1--------->horse\n2---------->human\n');
D1 = dir('D:\mywork\ML\projects\HorseOrHuman\horse-or-human\horses\*.png');

D2 = dir('D:\mywork\ML\projects\HorseOrHuman\horse-or-human\humans\*.png');

Z_human_main=[Z_human_main];
for counts=1:10
i=floor(100*rand(1)+1);
temp=imread(strcat(D2(i).folder,'\',D2(i).name));
imshow(temp);
x=Z_human_main(i,:);
pred=predict(Theta1, Theta2, x);
fprintf('\nthis is %d\n',pred);
pause;

end

for counts=1:10
i=floor(100*rand(1)+1);
temp=imread(strcat(D1(i).folder,'\',D1(i).name));
imshow(temp);
x=Z_horse_main(i,:);
pred=predict(Theta1, Theta2, x);
fprintf('\nthis is %d\n',pred);
pause;

end




