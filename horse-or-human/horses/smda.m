function X = preProcessingHorseImages()
X=[];


D = dir('*.png');
for i = 1:numel(D)
temp=imread(D(i).name);
temp=temp(:)';
  X=[X;temp];
end

imshow(X(2));
temp=reshape(X(2),300,300,3);
end