function preds = predict_net(trained_net,Xte)
    [m,d] = size(Xte);
    num_layers = numel(trained_net.W);
    a_h = Xte.';
    for h = 1:num_layers
        act_func = trained_net.A{h};
        W = trained_net.W{h};
        b = trained_net.B{h};
        z_h = W*a_h + b*ones(1, m);
        a_h = act_func(z_h);
    end
    preds = a_h;
end
