function Z =randomShuffle(Z)
fprintf('shuffling')
for i=1:20

fprintf('.')
Z=Z(randperm(size(Z,1)),:);


end
end