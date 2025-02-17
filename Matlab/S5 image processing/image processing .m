%Image loading, manipulation, and processing tasks, including basic operations like image display, manipulation, and conversion, as well as more complex tasks like vectorization and handling multiple images from a folder.

pt = imread('pt.bmt')
imshow(pt)

x = imread('x.bmp')

x=[1 1 1, 0 1 0, 0 1 0, 0 1 0];
pchar(x,4,3,1)

x1=[1 0 0 0 1; 0 1 0 1 0; 0 0 1 0 0; 0 1 0 1 0; 1 0 0 0 1]
x2=mbplr(x)
x3=mkvec(x2)

xp=mvcec(x2);
pchar(xp, 5, 5)
pchar(xp, 5, 5, 1)

imshow(x1)
imshow(mbplr(x1))
imshow(-mbplr(x1))

x4=corrupt(x3,10);
pchar(x4,5,5,1)

c2=imread(‘sym2.bmp');
imshow(c2)

c1=imread('sym1.bmp');
c1v=mkvec(c1);
c1b=-mbplr(c1v);
pchar(c1b, 80, 80)

c3=imread('sym3.bmp');
c3v=mkvec(c3);
c3b=-mbplr(c3v);
c3c=corrupt(c3b, 10);
pchar(c3c, 80, 80)
imshow(reshape(-c3c,80,80)’)




image=imread('monalisa.png');
imshow(image)
binimage = imbinarize(image);
imshow(binimage)


image2=imread('lion.png’);
igray=rgb2gray(image2);
ibw=imbinarize(igray);
imagen=double(image2)+100*randn(size(image2));
imagen=uint8(imagen);
imshow(imagen)


filelist=dir('*.png')
filelist(1).name
filelist(1).folder
fullfile(filelist(1).folder, filelist(1).name)
im1=imread(fullfile(filelist(1).folder, filelist(1).name));


%this part is for reading png images from folders and arranging input and output target vectors 

folder='mnist_ar\testing\';
images_tst = [];
labels_tst = [];
for dgt=0:9,
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
end;
end;