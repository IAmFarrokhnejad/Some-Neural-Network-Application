% This script showcases a random selection of 9 test digits, along with
% their actual classifications and the predicted classifications assigned
% by the trained model
%
% To ensure the network output for all test samples has been calculated,
% please run the test_digits.m script beforehand.
%
% Ahmet Rizaner, November 2024

% Convert the classes of the network output and actual targets to numbers
% between 0 and 9
tsti=vec2ind(labels_tst)-1;
tsto=vec2ind(YTst)-1;

% Randomly permute the samples
sInx= randperm(size(images_tst, 2));

% "Showcase 9 randomly selected test digit images"
figure(3);
for inx=1:9
subplot(3,3,inx);
imshow(reshape(images_tst(:, sInx(inx)), 28, 28)); 
title([num2str(tsti(sInx(inx))), ' : ', num2str(tsto(sInx(inx)))]);
end
    
% In case an error occurs during classification, assuming that digit
% number 6 was incorrectly classified, you can utilize the following
% code to examine the actual network output:
% YTst(:,sInx(6))

