% This script is designed to evaluate the performance of a multilayer
% perceptron (MLP) architecture that was trained to recognize handwritten
% digits from PNG images. The confusion matrix will be plotted to visualize
% the classification performance, and the accuracy of the test samples
% will be calculated.
%
% This MATLAB script assumes the neural network has already been trained
% and saved as "mlp_model.mat". If the trained network is not available,
% run the "train_digits.m" script to train the model.
%
% Ahmet Rizaner, November 2024

% Clear all variables
clear;

% Load the pre-trained neural network model (net) from the MAT file.
 load('mlp_model.mat');

% Import test digits and corresponding target outputs
% images_tst: test inputs, labels_tst: corresponding targets
test_data_import;

% Calculate the output of the network for the test samples
YTst=net(images_tst);

% Calculate and plot the confusion matrix for the test data
figure(2); plotconfusion(labels_tst, YTst);
title('Testing - Confusion Matrix')

% Percentage of correctly classified test samples - Accuracy (%)
[c,cm] = confusion(labels_tst, YTst);
fprintf('Test Accuracy : %.2f %%\n', 100*(1-c));