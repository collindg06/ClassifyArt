% This code takes an input image and predicts the classification
load('net.mat'); % Run this line to use the pretrained or latest trained model

% gets input and preprocesses image
disp("Jpg and png files are accepted. Note that the image should be in the directory your MatLab is in.")
image = input("Input image file name (and directory if needed) inside quotations: ");
image = imread(image);
image = augmentedImageDatastore([224 224],image,'ColorPreprocessing', 'gray2rgb');

% determine classification of image
class = classify(train, image);

% print out answer
disp(strcat("This image is ", string(class)))


