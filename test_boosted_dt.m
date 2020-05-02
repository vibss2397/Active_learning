function [ypred, yprob, yprob_raw] = test_boosted_dt(Xte, alpha, DTCell, learner)
    nb = size(alpha, 1);
    if(learner=="cnn")
        n = size(Xte,4);
    else
        n = size(Xte, 1);
    end
    yprob = zeros(n, 1);
    yprob_raw = zeros(n, nb);
    for i = 1:nb
      if(learner=="linear")
        answ = predict(DTCell{i}, Xte);
      elseif(learner=="nn")
         probs = predict_net(DTCell{i}, Xte);
         probs = probs.';
         answ = round(probs);
      elseif(learner=="cnn")
        [answ, probs] = classify(DTCell{i}, Xte);
      end
        answ = double(string(answ));
        yprob_raw(:, i) = answ;
        yprob = yprob + alpha(i)*answ;
    end
    ypred = sign(yprob);
        
    