function Z =randomShuffle(Z)
fprintf('\n\nshuffling')
for i=1:20

fprintf('.')
Z=Z(randperm(size(Z,1)),:);
Z=Z(randperm(size(Z,1)),:);
Z=Z(randperm(size(Z,1)),:);
Z=Z(randperm(size(Z,1)),:);



end
end