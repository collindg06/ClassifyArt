% This code tests ArtClassification.m and ArtClassificationTrain.m

% Ensure the net.mat file and art folder are in your current directory
% This part of the code trains and tests a neural network that can decipher between real art and ai art
load('net.mat');

% create image data store of all pictures
store = imageDatastore("Art\",IncludeSubfolders=true,LabelSource="foldernames");

% seperate train/test data and resize
[storeTrain,storeTest] = splitEachLabel(store,.5);
storeTrainRe = augmentedImageDatastore([224 224],storeTrain,'ColorPreprocessing', 'gray2rgb');
storeTestRe = augmentedImageDatastore([224 224],storeTest, 'ColorPreprocessing', 'gray2rgb');

% preallocation and initialization
group = 1;
f1Score = zeros(2,1);

% loop for n-fold cross validation
while (group < 3)

    % train network depending upon the group
    opts = trainingOptions("sgdm");
    if (group == 1)

        [train,labels] = trainNetwork(storeTrainRe, lgraph_1, opts);
        % predict image classification
        predictions = classify(train, storeTestRe);
        actual = storeTest.Labels;

    elseif (group == 2)

        [train,labels] = trainNetwork(storeTestRe, lgraph_1, opts);
        % predict image classification
        predictions = classify(train, storeTrainRe);
        actual = storeTrain.Labels;

    end

    % comparing the correct labels with the predictions
    correct = (predictions == actual);

    % preallocation and initialization
    correctLabels = strings(1000,1);
    falseLabels = strings(1000,1);
    counterCorrect = 1;
    counterFalse  = 1;

    % loop to get correct and false labels
    for x = 1 :length(predictions)
        if (correct(x) == 1)
            correctLabels(counterCorrect) = predictions(x);
            counterCorrect = counterCorrect + 1;
    
        elseif (correct(x) == 0)
            falseLabels(counterFalse) = predictions(x);
            counterFalse = counterFalse + 1;
        end
    end 

    % get the amount of correct predictions
    numCorrect = nnz(correct);

    % calculate true positive, false positive, true negative, and false negative
    truePositive = nnz(correctLabels == "Real");
    falsePositive = nnz(falseLabels == "Real");
    trueNegative = nnz(correctLabels == "Ai");
    falseNegative = nnz(falseLabels == "Ai");

    % calculate precision, recall, and F1 score
    precision = truePositive / (truePositive + falsePositive) ;
    recall = truePositive / (truePositive + falseNegative);
    f1Score(group) = 2 * ((precision * recall) / (precision + recall));

    % print out
    disp (strcat("Group " , num2str(group)))
    disp (strcat("Precision = " , num2str(precision)));
    disp (strcat("Recall = " , num2str(recall)));
    disp (strcat("F1 score = " , num2str(f1Score(group))));
    
    group = group +1;

end

% calculate and display the average f1 score
averagePerformance = (f1Score(1) + f1Score(2)) / 2;
disp(strcat("Average performance of model in terms of F1 score: ", num2str(averagePerformance)))
%% 
% This part of code takes input image and predicts the classification

% gets input and preprocesses image
disp("Jpg and png files are accepted. Note that the image should be in the directory your MatLab is in.")
image = "aiimage.jpg";
image = imread(image);
image = augmentedImageDatastore([224 224],image,'ColorPreprocessing', 'gray2rgb');

% determine classification of image
class = classify(train, image);

% print out answer
disp(strcat("This image is ", string(class)))


