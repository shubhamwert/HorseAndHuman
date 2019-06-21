function [U, S] = pca(X)
temp=double(X);
size(X);

[ m , n ] = size(temp);
fprintf('\n\nCalculating covariance matrix\n\n');
temp2=(temp'*temp)/m;
[U,S,D]=svd(temp2);
fprintf('\n\npca done\n\n');