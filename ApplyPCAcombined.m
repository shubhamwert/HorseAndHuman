function [Z_horse,Z_human] = ApplyPCAcombined()


%%Load data from folders Humans And Horses
data_horse_X=double(LoadHorseImages());
data_human_X=double(LoadHumanImages());

data_horse_X=data_horse_X(:,1:20:end);             %%Note :: to match the octave index dimensions of the image is reduced
data_human_X=data_human_X(:,1:20:end);
fprintf('\nSize of Horse Data\n');
ho_l=size(data_horse_X,1);
size(data_horse_X)

fprintf('\nSize of Human Data\n');
hu_l=size(data_human_X,1);

size(data_human_X)


X=[data_horse_X;data_human_X];
fprintf('Applyig feature normalization')
[X_norm,X_mean,X_sigma]=normalizeFeatures(X/255);

fprintf('\nRunnign PCA\n');
[U,S]=pca(X_norm);

K=selectK(S);

fprintf('\nselected K is %d\n',K);

Z= projectData(X_norm, U, K);

fprintf('\nsaving Z params ...\n')
csvwrite('Z_combined.csv',Z);




