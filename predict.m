function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

 
p = zeros(size(X, 1), 1);


p1=sigmoid([ones(m,1) X]*Theta1');
p2=sigmoid([ones(m,1) p1]*Theta2');

[a,p]=max(p2,[],2);


end
