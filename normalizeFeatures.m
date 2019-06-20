%Code is inspired from ML course by Andrew Ng Coursera 
function [X_norm, mu, sigma] = normalizeFeatures(X)

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end
