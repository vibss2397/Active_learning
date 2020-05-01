function [cnet, info] = cnn(XTrain, YTrain, t)
    layers = [
           imageInputLayer([64 64 1])
           
           convolution2dLayer(3,16,'Padding','same')
           reluLayer
           maxPooling2dLayer(2,'Stride',2)
           
           convolution2dLayer(3,32,'Padding','same')
           reluLayer
           maxPooling2dLayer(2,'Stride',2)
           
%{           
           convolution2dLayer(3,64,'Padding','same')
           reluLayer
           maxPooling2dLayer(2,'Stride',2)
           
         
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
 
        convolution2dLayer(3,64,'Padding','same')
        reluLayer
 %}    
        convolution2dLayer(3,32,'Padding','same')
        reluLayer

        fullyConnectedLayer(128, 'name', 'fc1')
        reluLayer
        
        fullyConnectedLayer(2, 'name', 'fc3')
        softmaxLayer
        classificationLayer
        ];

    options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'gpu',...
    'MaxEpochs',t,...
    'InitialLearnRate',1e-3, ...
    'Verbose',false);    
    YTrain = categorical(YTrain);
    [cnet, info] = trainNetwork(XTrain, YTrain,layers,options);
end