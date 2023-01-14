# Optical-Character-Recognition (OCR) using Fourier Transform (FT)

Fourier Transform (FT) and Euclidean distance-based OCR method is implemented. It extracts frequency features of a character such as digits and alphabet. MATLAB has been used as an implementation tool for different preprocessing techniques like cropping, resizing, Edge detection, and filtering purposes.


### [Dataset](https://github.com/Lakshay-Bansal/Optical-Character-Recognition-OCR-using-Fourier-Transform/tree/main/Dataset)    [Codes](https://github.com/Lakshay-Bansal/Optical-Character-Recognition-OCR-using-Fourier-Transform/tree/main/Codes)    [Result](https://github.com/Lakshay-Bansal/Optical-Character-Recognition-OCR-using-Fourier-Transform/tree/main/Result)


## Algorithm Steps

* Step-1: Dataset of the handwritten images.

* Step-2: The images are then downsized to a 64*64 pixel size. The goal is to reduce the amount of storage space needed to store all images. That also reduce the number of computations required to train and evaluate a model, which will shorten the program's runtime.

* Step-3: The image is then processed using a median filter followed by edge detection.

* Step-4: The FT of the preprocessed and resized input images is computed in this stage.

* Step-5: This phase involves calculating the Euclidean distance between two Fourier Transform of the train and test images respectively.

* Step-6: Euclidean distance between FT of each train image w.r.t FT of each available test image and stored in the array.

* Step-7: The index of array that corresponds to the shortest Euclidean distance is used to represent the predicted image.

**At last models accuracy is calculated by number of correct prediction and the number of test subject.**




## Additional Resources

1. [Overview](https://github.com/Lakshay-Bansal/Optical-Character-Recognition-OCR-using-Fourier-Transform/wiki)
2. [References](https://github.com/Lakshay-Bansal/Optical-Character-Recognition-OCR-using-Fourier-Transform/wiki/References)

