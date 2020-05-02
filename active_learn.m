%assuming they are in the same folder, the link to the dataset in .mat is
%different and given in report
load('dataset.mat')
%Xtrain contains pixel values from 0 to 255, so normalizing them,
xtrain_norm = reshape(Xtrain, [64, 64, 1, 22000]);
%use this with linear classifier
% xtrain_norm = reshape(Xtrain, [22000, 4096]);

xtrain_norm = (xtrain_norm - 127.5)/127.5;

%converting where y is -1 to 0, to be used by nn, also as is evident as is later
%used in this algorithm
Ytrain2(Ytrain==-1) = 0;

%number of samples, change to 1 if using linear classifier or nn
n = size(xtrain_norm, 4);

%Ytrain2 is shuffled, randomly sampling will give a split
%between both the classes
labelled_indices = randperm(n,1000);

all_indices = (1:n);
unlabelled_indices = setdiff(all_indices, labelled_indices);



%iterations for active learning
n_active = 30;
n_adaboost = 10;
%chosing top k queries
k = 100;
acc = zeros(n_active, 1);
for i=1:n_active
    disp(['active learning, round : ', num2str(i)])
    %setting up labelled and unlabelled data, replace it with xtrain_norm(labelled_indices, :) if using nn or linear
    Xtrain_labelled = xtrain_norm(:, :, :, labelled_indices);
    ytrain_labelled = Ytrain(labelled_indices);
    Xtrain_unlabelled = xtrain_norm(:, :, :,unlabelled_indices);
    ytrain_unlabelled = Ytrain(unlabelled_indices);
    
    [alpha, learnerCell, tr_err] = train_boosted_dt(Xtrain_labelled, ytrain_labelled, n_adaboost, "cnn");
    %using ypred for lower confidence sampling 
    [ypred, yprob, yprob_raw] = test_boosted_dt(Xtrain_unlabelled, alpha, learnerCell, "cnn");
    %signum only returns labels as 1 or -1 converting all -1's to 0
    ypred_temp = ypred;
    ypred_temp(ypred_temp==-1) = 0;
    
    acc(i) = mean(ypred==ytrain_unlabelled);
    disp(['active learning round', num2str(i)]);
    disp(acc(i));
    %sum of all alphas which are ones and zeros,other alphas cancel out
    alpha_ones = yprob_raw*alpha;
    alpha_zeros = (1 - yprob_raw)*alpha;
    alpha_diff = alpha_ones - alpha_zeros;
    [top_alpha_diff, index] = mink(alpha_diff, k);
    
    %setting up labelled and unlabelled indices for the next iterations
    labelled_indices = [labelled_indices, index.'];
    unlabelled_indices = setdiff(unlabelled_indices, index.');
    %disp(size(labelled_indices,2));
    %disp(size(unlabelled_indices,2));
end    

    