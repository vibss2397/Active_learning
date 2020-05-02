load('dataset.mat')
%Xtrain contains pixel values from 0 to 255, so normalizing them,
%converting where y is -1 to 0, to be used by nn
%Xtrain2 = (Xtrain2 - 127.5)/127.5;
%Ytrain2(Ytrain2==-1) = 0;

n = size(xtrain_norm, 4);

%Ytrain2 is shuffled, randomly sampling will give a split
%between both the classes
labelled_indices = randperm(n,1000);

all_indices = (1:n);
unlabelled_indices = setdiff(all_indices, labelled_indices);



%iterations for active learning
n_active = 15;
n_adaboost = 15;
%chosing top k queries
k = 1000;
acc = zeros(n_active, 1);
for i=1:n_active
    %setting up labelled and unlabelled data
    Xtrain_labelled = xtrain_norm(:, :, :, labelled_indices);
    ytrain_labelled = Ytrain2(labelled_indices);
    Xtrain_unlabelled = xtrain_norm(:, :, :,unlabelled_indices);
    ytrain_unlabelled = Ytrain2(unlabelled_indices);
    
    [cnet, info] = cnn(Xtrain_labelled, ytrain_labelled, 15);
    %using ypred for lower confidence sampling 
    [ypred, yprob]=  classify(cnet, Xtrain_unlabelled);
    ypred_diff = abs(yprob(:, 1) - yprob(:, 2));
    %signum only returns labels as 1 or -1 converting all -1's to 0
    ypred_temp = double(string(ypred));
    ypred_temp(ypred_temp==-1) = 0;
    
    acc(i) = mean(ypred_temp==ytrain_unlabelled);
    disp(acc(i));
    %sum of all alphas which are ones and zeros,other alphas cancel out
    [top_ypred, index] = mink(ypred_temp, k);
    %setting up labelled and unlabelled indices for the next iterations
    labelled_indices = [labelled_indices, index.'];
    unlabelled_indices = setdiff(unlabelled_indices, index.');
    disp(size(labelled_indices,2));
    disp(size(unlabelled_indices,2));
end    

    
