%% Generate Data
% To save the matrix having image as its column vectors and another matrix to 
% store their Fourier Transform.

clear all
clc
% There are 55 images of 0-9, A-Z, a-z handwritten images
files = imageDatastore(strcat('D:\Lakshay\M.Tech\Sem 1\EE608_Digital Image Processing\DIP Project\7_Best_62Classes\Img'), "FileExtensions",'.png').Files;
nFiles = size(files)    % we require just 55 * (10 + 26) = 1980 files from this readed files
%% Generating Train Matrix

no_of_trainImg = 40;    % We have a 55 image of each character
dim_img = 64*64;
train_imgMat = zeros(dim_img,no_of_trainImg*36);    % 36 Character 0-9 and A-B
train_img_FT_Mat = zeros(dim_img,no_of_trainImg*36);

for alpha = 1:36    % 10 digits + 26 Upper case aplhabet
    for n = 1:no_of_trainImg
        imgNum = 55*(alpha-1)+n;    % As 55 image for each alphabet 
        img = im2gray(imread(files{imgNum}));
        imgSmall = imresize(img, [64 64]);  % Resizing the image to small size
        TimgFil = medfilt2(imgSmall); % To improve the quality of resize image
        img_edge = edge(TimgFil, "canny");
        [row, col]= size(img_edge);
        
        train_MatColNum = no_of_trainImg*(alpha-1)+n;
        train_imgMat(:, train_MatColNum) = reshape(img_edge, row*col , 1);
        
        %% Image FT
        ftImg = fft2(img_edge);
        ftImg_Norm = abs(ftImg)*255/max(max(abs(ftImg)));   % FT of Image being Normalised
        train_img_FT_Mat(:, train_MatColNum) = reshape(fftshift(ftImg_Norm), row*col , 1);
    end
end
%%
% img = im2gray(imread(files{100}));
% imgSmall = imresize(img, [64 64]);  % Resizing the image to small size
% img_edge = edge(imgSmall, "canny");
% figure; imshow(img)
% figure; imshow(imgSmall)
% figure; imshow(img_edge)
% imgFil = medfilt2(imgSmall);
% figure; imshow(imgFil)
% img_edge2 = edge(imgFil, "canny");
% figure; imshow(img_edge2)
%% Generating Test Matrix

% From 55 images of each alphabet, 40 for taining. Hence 15 for testing
% purpose 
no_of_testImg = 15; 
dim_img = 64*64;
test_imgMat = zeros(dim_img,no_of_testImg*36);   % 36 Character 0-9 and A-B
test_img_FT_Mat = zeros(dim_img,no_of_testImg*36);

for alpha = 1:36    % 10 digits + 26 Upper case aplhabet
    for n = 1:no_of_testImg
        imgNum = 55*(alpha-1) + no_of_trainImg + n;    % As 55 image for each alphabet 
        Timg = im2gray(imread(files{imgNum}));
        TimgSmall = imresize(Timg, [64 64]);  % Resizing the image to small size
        TimgFil = medfilt2(TimgSmall); % To improve the quality of resize image
        Timg_edge = edge(TimgFil, "canny");
        [row, col]= size(Timg_edge);
        
        test_MatColNum = no_of_testImg*(alpha-1)+n;
        test_imgMat(:, test_MatColNum) = reshape(Timg_edge, row*col , 1);
        
        %% Image FT
        ftImg = fft2(Timg_edge);
        ftImg_Norm = abs(ftImg)*255/max(max(abs(ftImg)));   % FT of Image being Normalised
        test_img_FT_Mat(:, test_MatColNum) = reshape(fftshift(ftImg_Norm), row*col , 1);
    end
end
%%
save imgEdgeMatrix.mat train_imgMat test_img_FT_Mat test_imgMat train_img_FT_Mat