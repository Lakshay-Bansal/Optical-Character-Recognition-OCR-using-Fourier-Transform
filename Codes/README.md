## About Code File

1) **dataGenerator_EdgeImg.mlx / dataGenerator_EdgeImg.m** - This program is being used to generate train and test matrix, where column correspond to the single image being store as vector

- The program store the generated matrix in "imgEdgeMatrix.mat" file. This is being done to reduce the run time of a program Char_recog -Main.mlx / Char_recog -Main.m and to have modularity in our code. We are using 40 images of each class (0-9 and A-Z) to train the model and 15 image as a test cases. For reducing the computation time and for introducing uniformity in the dimension of all images we are rezing each image to 64*64 dimension. 

- The following matix are generated :

i) train_imgMat - Matrix of Train Images where column correspond to image represented as vector. Dimension is 4096*1440. 
( As we resizing the image into 64*64 dimension. Therefore number of rows = 64*64 = 4096. No. of train image are 40*(10+26) = 1440. )

ii) train_img_FT_Mat - Matrix of Fourier Transform of train Images where column correspond to image vector. Dimension is 4096*1440 same as of train_imgMat matrix. 
( Fourier Transform is calculated on the image which is obtained as a result of resize 64*64 image being filter by median filter followed by edge detection using Canny Edge detection method. )

iii) test_imgMat - Matrix of Test Images where column correspond to image vector. Dimension is 4096*540. 
( As we resizing the image into 64*64 dimension. Therefore number of rows = 64*64 = 4096. No. of testimage are 15*(10+26) = 540 which correspond to colums of test_imgMat matrix. )

iv) test_img_FT_Mat - Matrix of Fourier Transform of test images where column correspond to image vector. Dimension is 4096*540 same as of test_imgMat matrix. 
( Fourier Transform is calculated on the image which is obtained as a result of resize 64*64 image being filter by median filter followed by edge detection using Canny Edge detection method. )


2) **Char_recog -Main.mlx / Char_recog -Main.m** - This is the main file to be run to predict the image for corresponding test images. It loads the "imgEdgeMatrix.mat" file for its operation.

- This also gives the glimpse of the original clases available in the dataset. It plots the 0-9 classes along with their FT. Also A-Z classes and its FT in a seperated grid image.
- Then it plot the images present in the train_imgMat matrix one for each classes and their FT which are stored in train_img_FT_Mat matrix.
- Then the "charDict" named dictionary is created to store classes as key value pair.
- Evaluation of model is done on the test_imgMat images followed by the accuracy calculateion.
- At last we test the model on the unknown dataset created manually.


### Results :

* **imgEdgeMatrix.mat** - As explained above as well this file is obatained from dataGenerator_EdgeImg.mlx / dataGenerator_EdgeImg.m.

* **Char_recog - Results.pdf** - It have the code and output obtained from running the Char_recog -Main.mlx / Char_recog -Main.m file.


###### **Note** - Both '.m' and '.mlx' are the same program, however '.mlx' provide a better user interface while executing a code much like '.ipynb' python jupyter notebooks. 
