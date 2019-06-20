function X = LoadHorseImages(count)
X=[];


D = dir('D:\mywork\ML\projects\HorseOrHuman\horse-or-human\horses\*.png');
fprintf('\nhorse data \n\nloading');

for i = 1:numel(D)
temp=imread(strcat(D(i).folder,'\',D(i).name));
temp=temp(:)';
  X=[X;temp];
  fprintf('.');
end


 
 end