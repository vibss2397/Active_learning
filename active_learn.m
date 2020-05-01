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
n_active = 5;
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
    
    [alpha, learnerCell, tr_err] = train_boosted_dt(Xtrain_labelled, ytrain_labelled, n_adaboost, "cnn");
    %using ypred for lower confidence sampling 
    [ypred, yprob, yprob_raw] = test_boosted_dt(Xtrain_unlabelled, alpha, learnerCell, "cnn");
    %signum only returns labels as 1 or -1 converting all -1's to 0
    ypred_temp = ypred;
    ypred_temp(ypred_temp==-1) = 0;
    
    acc(i) = mean(ypred==ytrain_unlabelled);
    disp(acc(i));
    %sum of all alphas which are ones and zeros,other alphas cancel out
    alpha_ones = yprob_raw*alpha;
    alpha_zeros = (1 - yprob_raw)*alpha;
    alpha_diff = alpha_ones - alpha_zeros;
    [top_alpha_diff, index] = maxk(alpha_diff, k);
    %setting up labelled and unlabelled indices for the next iterations
    labelled_indices = [labelled_indices, index.'];
    unlabelled_indices = setdiff(unlabelled_indices, index.');
    disp(size(labelled_indices,2));
    disp(size(unlabelled_indices,2));
end    

    