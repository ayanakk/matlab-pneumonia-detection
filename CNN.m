rootFolder = '..Dataset/chest_xray';
trainFolder = fullfile(rootFolder, 'train');
testFolder = fullfile(rootFolder, 'test');

%ImageDatastore with Preprocessing
trainDatastore = imageDatastore(trainFolder, ...
    'LabelSource', 'foldernames', 'IncludeSubfolders', true);
testDatastore = imageDatastore(testFolder, ...
    'LabelSource', 'foldernames', 'IncludeSubfolders', true);
trainDatastore.ReadFcn = @(filename) preprocessImage(imread(filename));
testDatastore.ReadFcn = @(filename) preprocessImage(imread(filename));

layers = [
    imageInputLayer([200 200 3], 'Name', 'input', 'Normalization', 'none')

    convolution2dLayer(15, 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    convolution2dLayer(9, 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')

    convolution2dLayer(5, 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4')

    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu_fc2')

    fullyConnectedLayer(2, 'Name', 'fc_output') 
    sigmoidLayer('Name', 'sigmoid')
    classificationLayer('Name', 'output')  
];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valAugmented, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'MiniBatchSize', 32);

net = trainNetwork(trainDatastore, layers, options);

YPred = classify(net, testDatastore); 
YTest = testDatastore.Labels; 
% Calculate accuracy
accuracy = sum(YPred == testDatastore.Labels) / numel(testDatastore.Labels);
disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);

% Confusion Matrix
confMat = confusionmat(testDatastore.Labels, YPred);
disp('Confusion Matrix:');
disp(confMat);
figure;
confusionchart(confMat, {'Normal', 'Pneumonia'}, 'Title', 'Confusion Matrix');
%% 
% Confusion Matrix 
confMat = confusionmat(YTest, YPred);
TP = confMat(2, 2);  
TN = confMat(1, 1);  
FP = confMat(1, 2);  
FN = confMat(2, 1);  

% Precision
precision = TP / (TP + FP);
disp(['Precision: ', num2str(precision)]);

% Recall
recall = TP / (TP + FN);
disp(['Recall: ', num2str(recall)]);

% F1-Score
f1Score = 2 * (precision * recall) / (precision + recall);
disp(['F1-Score: ', num2str(f1Score)]);



%% Image Preprocessing Function
function imgOut = preprocessImage(img)
    imgResized = imresize(img, [200, 200]);

    % Intensity normalization
    imgResized = double(imgResized);
    imgMin = min(imgResized(:));
    imgMax = max(imgResized(:));
    imgNormalized = (imgResized - imgMin) / (imgMax - imgMin);

    % Gaussian filtering 
    imgGaussianFiltered = imgaussfilt(imgNormalized, 1);

    % CLAHE (Contrast Limited Adaptive Histogram Equalization)
    imgCLAHE = adapthisteq(imgGaussianFiltered, 'ClipLimit', 0.01, 'NumTiles', [8 8]);

    if size(imgCLAHE, 3) == 1
        imgOut = repmat(imgCLAHE, [1, 1, 3]);
    else
        imgOut = imgCLAHE;
    end
end
