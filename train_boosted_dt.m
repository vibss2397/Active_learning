function [alpha, learnerCell,tr_err] = train_boosted_dt(Xtr, ytr, T, learner)
    % a = fitctree(Xtr, ytr, 'MaxNumSplits', 3);
    learnerCell = {};
    tr_err = zeros(T,1);
    alpha = zeros(T, 1);
    ytr2 = double(string(ytr));
    if(learner=="tree")
        n = size(Xtr, 1);
    elseif(learner=="nn")
        n = size(Xtr, 1);
    elseif(learner=="cnn")
        n = size(Xtr, 4);
    end
    
    d = size(Xtr, 2);
    dt = zeros(n, 1)+1/n;
    prev_error = 1;
    for t = 1:T
      disp(['ada', num2str(t)]);
      if(learner=="tree")
        dt_temp = fitctree(Xtr, ytr2, 'MaxNumSplits',3,'Weights', dt); 
      elseif(learner=="nn")
        indices = datasample((1:n), n, 'Weights', dt);       
        [dt_temp, nn_loss] = neural_train(Xtr(indices, :), ytr(indices), [256, 128], 200, 0.001);  
      elseif(learner=="cnn")
        indices = datasample((1:n), n, 'Weights', dt);
        [dt_temp, info] = cnn(Xtr(:, :, :, indices), ytr(indices), 10);
      end
      learnerCell{t} = dt_temp;
      if(learner=="tree")
        answ = predict(dt_temp, Xtr);
      elseif(learner=="nn")
          probs = predict_net(dt_temp, Xtr);
          probs = probs.';
          answ = round(probs);
      elseif(learner=="cnn")
        answ = classify(dt_temp, Xtr);
      end
      answ = double(string(answ));
      
      one_err = answ~=ytr2;
      epsilon_t = dt.'*one_err;
      alpha(t) = 0.5*log((1/epsilon_t)-1);
      dt = dt.*(exp(-alpha(t)*(answ.*ytr2)));
      
      sum_dt = sum(dt);
      dt = dt/sum_dt;
      [final_outputs_tr, ~, ~] = test_boosted_dt(Xtr, alpha(1:t), learnerCell, learner);
      tr_err(t) = mean(final_outputs_tr~=ytr2);
      disp(tr_err(t));
    end
        
    
    
    