% This script is designed to train a multilayer perceptron (MLP)
% architecture for recognizing handwritten digits from PNG images.
% The MLP is a type of artificial neural network that can be used to
% classify data. In this case, the MLP will be trained to classify images
% of handwritten digits into the ten digits (0-9). Since the number of
% training samples is too large, we will use a common technique called
% mini-batch training. Mini-batch training involves dividing the training
% data into smaller batches and then training the network on one batch at
% a time. This can be more efficient than training the network on the
% entire dataset at once, as it can reduce the amount of memory required
% and make it easier to parallelize the training process.
%
% Ahmet Rizaner, November 2024

% Clear all variables
clear;

% import training inputs and corresponding target outputs
% images_tr: training inputs, labels_tr: corresponding targets
train_data_import; 

% The number of epochs to use for training
numEpochs=50;

% Set batch size,  number of training examples that will be processed in
% each iteration
batchSize = 500;

% Design a MLP with 100 hidden units
net = patternnet(100);

% Set the trainig parameters
net.trainParam.showCommandLine=true;
net.trainParam.showWindow=false;
net.trainparam.show=100; 

% All samples will be used for training
net.divideParam.trainratio=1.0;
net.divideParam.valratio=0.0;
net.divideParam.testratio=0.0;


for epoch = 1:numEpochs

    % Shuffle the data
    shuffledIndices = randperm(size(images_tr, 2));
   
    images_tr_shuffled = images_tr(:, shuffledIndices);
    labels_tr_shuffled = labels_tr(:, shuffledIndices);

    % Divide the data into mini-batches
    numBatches = floor(size(images_tr, 2) / batchSize);
    
    for batchNum = 1:numBatches

        disp(['Epoch : ', num2str(epoch), ' Batch : ', num2str(batchNum)])

        startIdx = (batchNum - 1) * batchSize + 1;
        endIdx = batchNum * batchSize;

        images_batch = images_tr_shuffled(:, startIdx:endIdx);
        labels_batch = labels_tr_shuffled(:, startIdx:endIdx);

        % Train the network on the mini-batch
        net=train(net,images_batch,labels_batch);
        
    end
end

% Save the trained neural network (net) as a MAT file for further use.
save('mlp_model.mat', 'net');

%Calculate and plot the confusion matrix for the training data
YTr=net(images_tr);
figure(1); plotconfusion(labels_tr, YTr);
title('Training - Confusion Matrix')
