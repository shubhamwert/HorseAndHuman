fprintf('\nreading combined Z\n');


Z_main = double(csvread('Z_combined.csv'));

fprintf('\nseprating Horse and Human Values\n');

Z_horse=Z_main(1:500,:);
Z_human=Z_main(501:1027,:);

fprintf('\nassinging Y\n ')
Z_horse=[Z_horse ones(size(Z_horse,1),1) zeros(size(Z_horse,1),1)];
Z_human=[Z_human zeros(size(Z_human,1),1) ones(size(Z_human,1),1)];

fprintf('\nCombining Data..\n')
Z=[Z_horse;Z_human];

fprintf('\nshuffling combined data....\n');

Z=randomShuffle(Z);

fprintf('\ncalling NN training system\n');
%%NN function 
[Theta1,Theta2]=TrainNN(Z);

fprintf('\ntesting real time\n1--------->horse\n2---------->human\n');
D1 = dir('D:\mywork\ML\projects\HorseOrHuman\horse-or-human\horses\*.png');

D2 = dir('D:\mywork\ML\projects\HorseOrHuman\horse-or-human\humans\*.png');


for i=1:7:49


temp=imread(strcat(D1(i).folder,'\',D1(i).name));
imshow(temp);
x=Z_main(i,:);
pred=predict(Theta1, Theta2, x);
fprintf('\nthis is %d\n',pred);
pause;


end
for i=1:7:50


temp=imread(strcat(D2(i).folder,'\',D2(i).name));
imshow(temp);
x=Z_main(i,:);
pred=predict(Theta1, Theta2, x);
fprintf('\nthis is %d\n',pred);
pause;


end