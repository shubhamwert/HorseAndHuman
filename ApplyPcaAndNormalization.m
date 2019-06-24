function [Z_horse,Z_human] = ApplyPcaAndNormalization()

clear; close all; clc; 

%%Load data from folders Humans And Horses
data_horse_X=double(LoadHorseImages());
data_human_X=double(LoadHumanImages());

data_horse_X=data_horse_X(:,1:20:end);             %%Note :: to match the octave index dimensions of the image is reduced
data_human_X=data_human_X(:,1:20:end);
fprintf('\nSize of Horse Data\n');

size(data_horse_X)

fprintf('\nSize of Human Data\n');

size(data_human_X)



%%featureNormalize both data
fprintf('\nNormalizing features for Horse Data\n');

[data_horse_X_norm,data_horse_X_mean,data_horse_X_sigma]=normalizeFeatures(data_horse_X/255);

fprintf('\nNormalizing features for Human Data\n');

[data_human_X_norm,data_human_mean,data_human_sigma]=normalizeFeatures(data_human_X/255);

%%dataset is too large reduce the dimensions or factors

fprintf('\nRunnign PCA for horses\n');

[U_horse,S_horse]=pca(data_horse_X_norm);
fprintf('\nRunnign PCA for humans\n');

[U_human,S_human]=pca(data_human_X_norm);
data_horse_X=[];
data_human_X=[];
K_horse=selectK(S_horse);
K_human=selectK(S_human);

K=max([K_horse K_human]);

fprintf('\n\nreducing dimensions for horse to %d\n',K );
Z_horse = projectData(data_horse_X_norm, U_horse, K);
fprintf('\nreducing dimensions for humans to %d\n\n',K)
Z_human = projectData(data_human_X_norm, U_human, K);
fprintf('\n\nPCA implementation completed \n \n');

fprintf('The projected data Z horse has a size of: ');
fprintf('%d ', size(Z_horse));
fprintf('\nsaving Human and horse data \n');

csvwrite('PCA_Horse_values.csv',Z_horse);
csvwrite('PCA_Human_values.csv',Z_human);

fprintf('\nsaving U params ...\n')
csvwrite('U_horse.csv',U_horse);
csvwrite('U_human.csv',U_human);

fprintf('\nsaving S params...\n')
csvwrite('S_horse.csv',S_horse);
csvwrite('S_human.csv',S_human);


end