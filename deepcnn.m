%% Dataset Path Setup
rootFolder = '../Dataset/new';
trainFolder = fullfile(rootFolder, 'train');
testFolder = fullfile(rootFolder, 'test');

% Define categories
categories = {'NORMAL', 'PNEUMONIA'};

% Create imageDatastore for train and test sets with a custom ReadFcn
trainImds = imageDatastore(fullfile(trainFolder, categories), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'ReadFcn', @validateAndPreprocess);
testImds = imageDatastore(fullfile(testFolder, categories), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'ReadFcn', @validateAndPreprocess);

%% Define the CNN Architecture with 52 Convolutional Layers
layers = [
    imageInputLayer([200 200 3], 'Name', 'input')
];

% Add 52 convolutional layers in groups, with pooling layers every few layers
for i = 1:52
    layers = [
        layers
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', sprintf('conv%d', i))
        batchNormalizationLayer('Name', sprintf('bn%d', i))
        reluLayer('Name', sprintf('relu%d', i))
    ];

    % Add a pooling layer every 13 layers to reduce spatial size
    if mod(i, 13) == 0
        layers = [
            layers
            maxPooling2dLayer(2, 'Stride', 2, 'Name', sprintf('pool%d', i/13))
        ];
    end
end

% Add fully connected, dropout, and classification layers
layers = [
    layers
    fullyConnectedLayer(512, 'Name', 'fc1')
    dropoutLayer(0.5, 'Name', 'dropout')
    reluLayer('Name', 'relu_fc')
    fullyConnectedLayer(2, 'Name', 'fc_output')  % Two classes: NORMAL and PNEUMONIA
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% Training Options (GPU Enabled)
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'ValidationData', testImds, ...
    'ValidationFrequency', 30, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu', ...  % Set to 'gpu' for GPU usage
    'OutputNetwork', 'best-validation-loss');

%% Train the Network
net = trainNetwork(trainImds, layers, options);

%% 
%% Save the Trained Model
modelFilename = 'deepcnn.mat';
save(modelFilename, 'net');
disp(['Model saved as ', modelFilename]);


%% Evaluate the Network on the Test Set
YPred = classify(net, testImds);  % Predicted labels
YTest = testImds.Labels;          % True labels

% Calculate test accuracy
testAccuracy = sum(YPred == YTest) / numel(YTest);
disp(['Test Accuracy: ', num2str(testAccuracy * 100), '%']);

% Print Confusion Matrix
confMatrix = confusionmat(YTest, YPred);
disp('Confusion Matrix:');
disp(confMatrix);

% Display confusion matrix as a heatmap
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');
%% 

%% Load the Trained Model
modelFilename = 'deepcnn.mat'; % Specify the model file name
rootFolder = '../Dataset/new';
testFolder = fullfile(rootFolder, 'test');
testImds = imageDatastore(fullfile(testFolder, categories), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'ReadFcn', @validateAndPreprocess);


YPred = classify(net, testImds);  % Predicted labels
YTest = testImds.Labels;          % True labels

% Calculate Test Accuracy
testAccuracy = mean(YPred == YTest) * 100;
disp(['Test Accuracy: ', num2str(testAccuracy), '%']);

% Print Confusion Matrix
confMatrix = confusionmat(YTest, YPred);
disp('Confusion Matrix:');
disp(confMatrix);

% Display confusion matrix as a heatmap
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');

%% Helper Function to Validate and Preprocess Images
function imgOut = validateAndPreprocess(filename)
    try
        % Read the image
        img = imread(filename);
        
        % Check if the image is empty or too small
        if isempty(img) || size(img, 1) < 2 || size(img, 2) < 2
            error('Invalid image dimensions.');
        end

        % Convert to grayscale if the image is RGB
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        % Normalize the intensity using Min-Max normalization
        img = double(img);
        imgMin = min(img(:));
        imgMax = max(img(:));
        img = (img - imgMin) / (imgMax - imgMin);

        % Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        img = adapthisteq(img, 'ClipLimit', 0.01, 'NumTiles', [8 8]);

        % Apply Gaussian filtering
        img = imgaussfilt(img, 2);

        % Convert grayscale to RGB
        imgOut = repmat(img, [1, 1, 3]);

        % Resize the image to 200x200
        imgOut = imresize(imgOut, [200 200]);
    catch
        fprintf('Skipping invalid image: %s\n', filename);
        imgOut = []; % Return empty for invalid images
    end
end
