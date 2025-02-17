
% Import Test Patterns and Output Classes
% This file imports the testing characters in a matrix named 'labels' 
% and their corresponding output classes in a matrix named 'classes


folder='mnist_ar\testing\';

% Create empty matrices to store the images and labels.
images_tst = [];
labels_tst = [];

for dgt=0:9
    
    filelist=dir([folder, num2str(dgt),'\*.png']);
    
    for smp=1:length(filelist)
    
% Get a list of all the PNG image files in the folder


fullFileName =fullfile(filelist(smp).folder, filelist(smp).name);

img = double(imread(fullFileName));
images_tst = [images_tst, img(:)];

   % Store the corresponding folder number in the labels matrix.
            dgt_mat=zeros(10,1);
            dgt_mat(dgt+1)=1;
            labels_tst = [labels_tst, dgt_mat];

    end
    
end

    