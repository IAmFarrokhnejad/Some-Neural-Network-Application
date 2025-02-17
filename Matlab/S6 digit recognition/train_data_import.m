
% Import Training Patterns and Output Classes
% This file imports the training characters in a matrix named 'labels_tr' 
% and their corresponding output classes in a matrix named 'classes_tr'.


folder='mnist_ar\training\';

% Create empty matrices to store the images and labels.
images_tr = [];
labels_tr = [];

for dgt=0:9
    
    filelist=dir([folder, num2str(dgt),'\*.png']);
    
    for smp=1:length(filelist)
    
% Get a list of all the PNG image files in the folder


fullFileName =fullfile(filelist(smp).folder, filelist(smp).name);

img = double(imread(fullFileName));
images_tr = [images_tr, img(:)];

   % Store the corresponding folder number in the labels matrix.
            dgt_mat=zeros(10,1);
            dgt_mat(dgt+1)=1;
            labels_tr = [labels_tr, dgt_mat];

    end
    
end

    