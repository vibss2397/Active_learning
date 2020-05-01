Xtrain2 = zeros(64, 64, 25000);
Ytrain2 = zeros(25000, 1);
counter = 0;
for a = 1:12500
   filename = ['train-re2/cat.' num2str(a-1,'%01d') '.jpg'];
   img = imread(filename);
   Xtrain2(:, :, a) = img;
   Ytrain2(a)= -1;
   disp(counter);
   counter = counter + 1;
   % do something with img
end

 for a = 1:12500
   filename = ['train-re2/dog.' num2str(a-1,'%01d') '.jpg'];
   img = imread(filename);
   Xtrain2(: , :, a+12500) = img;
   Ytrain2(a+12500)= 1;
   disp(counter);
   counter = counter + 1;
   % do something with img
end
idx = randperm(25000);
Xtrain = Xtrain2(:, :, idx(1:22000));
Ytrain = Ytrain2(idx(1:22000));
Xtest = Xtrain2(:, :, idx(22001:25000));
Ytest = Ytrain2(idx(22001:25000));
save dataset4.mat Xtrain Ytrain Xtest Ytest -v7.3;

