function [trained_net, nn_loss] = neural_train(Xtr, Ytr, nodes, T, lr)
    [m, d] = size(Xtr);
    if(size(Ytr, 1)==m)
        Ytr = Ytr.';
    end
    H = numel(nodes);
    W = {};b = {};a_h = {};z_h = {};
    dz = {};dw = {};db = {};
    activation = {};
    nn_loss = zeros(T, 1);
    sigmoid = @(x) 1./(1+exp(-x));
    lrelu = @(x) max(x, 0.02.*x);    
    d_sigmoid = @(x) sigmoid(x).*(1 - sigmoid(x));
    d_lrelu = @(x) 0.99.*(x>=0) + 0.02;
    
    bi_ce_loss = @(y, ypred) -(y.*log(ypred) + (1-y).*log(1-ypred));
    sq_loss = @(y, ypred) (y - ypred).^2;
    prev_nodes = d;
    
    for i = 1:H
        W{i} = normrnd(0, 1, nodes(i), prev_nodes);
        b{i} = zeros(nodes(i), 1);
        prev_nodes = nodes(i);
        activation{i} = lrelu;
    end
    W{H+1} = normrnd(0, 1, 1, prev_nodes);
    b{H+1} = zeros(1, 1);
    activation{H+1} = sigmoid;
    
    for t = 1:T
        lr = lr/sqrt(t);
        a_h{1} = Xtr.';
        for h = 1:H+1
            z_h{h} = (W{h}*a_h{h}) +b{h}*ones(1, m);
            if(h~=H+1)
                a_h{h+1} = sigmoid(z_h{h});
            end
        end
        
        yprob = sigmoid(z_h{H+1});
        ypred = round(yprob);
        %disp(yprob);
        %disp(['iteration ', num2str(t), ' loss ', num2str(sum(ypred~=Ytr, 2))]); 
        loss = sq_loss(Ytr, yprob);
        nn_loss(t) = mean(loss, 2);
        for h=H+1:-1:1
            if(h==H+1)
                dz{h} = (yprob - Ytr).*d_sigmoid(z_h{h});
            else
                dz{h} = (W{h+1}.'*dz{h+1}).*d_sigmoid(z_h{h});
            end
            dw{h} = dz{h}*a_h{h}.';
            db{h} = mean(dz{h}, 2);
        end
         for h = 1:H+1
            W{h} = W{h} - lr.*dw{h};
            b{h} = b{h} - lr.*db{h};
         end
        trained_net.W = W;
        trained_net.B = b;
        trained_net.A = activation;
    end