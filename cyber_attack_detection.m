% Extracting information from the Zip File
zip_ref = unzip('C:/Users/Taraneh/Documents/MATLAB/CSTR data files.zip', 'C:/Users/Taraneh/Documents/MATLAB');

% Reading the information using the scipy
Tr_attack = load('C:/Users/Taraneh/Documents/MATLAB/CSTR_train_attack.mat');
Tr_noise = load('C:/Users/Taraneh/Documents/MATLAB/CSTR_train_noise.mat');
Tr_normal = load('C:/Users/Taraneh/Documents/MATLAB/CSTR_train_normal.mat');
Te_attack = load('C:/Users/Taraneh/Documents/MATLAB/CSTR_test_attack.mat');
Te_noise = load('C:/Users/Taraneh/Documents/MATLAB/CSTR_test_noise.mat');
Te_normal = load('C:/Users/Taraneh/Documents/MATLAB/CSTR_test_normal.mat');

Train_attack = Tr_attack.CSTR_train_attack;
Train_noise = Tr_noise.CSTR_train_noise;
Train_Normal = Tr_normal.CSTR_train_normal;
Test_attack = Te_attack.CSTR_test_attack;
Test_noise = Te_noise.CSTR_test_noise;
Test_Normal = Te_normal.CSTR_test_normal;

% Checking the shape of extratced training and testing
disp("The size of the training dataset for Attack: " + string(size(Train_attack)));
disp("The size of the training dataset for Noise: " + string(size(Train_noise)));
disp("The size of the training dataset for Normal: " + string(size(Train_Normal)));
disp("The size of the testing dataset for Attack: " + string(size(Test_attack)));
disp("The size of the testing dataset for Noise: " + string(size(Test_noise)));
disp("The size of the testing dataset for Normal: " + string(size(Test_Normal)));

% Combining Training information
Combined_training = [Train_attack(:,1:end-1); Train_noise(:,1:end-1); Train_Normal(:,1:end-1)];
disp("The combined training set: " + string(size(Combined_training)));

Training_target = [Train_attack(:,end); Train_noise(:,end); Train_Normal(:,end)];
disp("The combined target training set: " + string(size(Training_target)));

% Combining Testing information
Combined_testing = [Test_attack(:,1:end-1); Test_noise(:,1:end-1); Test_Normal(:,1:end-1)];
disp("The combined target testing set: " + string(size(Combined_testing)));

Target_testing = [Test_attack(:,end); Test_noise(:,end); Test_Normal(:,end)];
disp("The combined target testing set: " + string(size(Target_testing)));

% if any(Training_target(:) < 0)
%     Training_target = abs(Training_target);
% end
% Creating an One Hot Encoder for the Target
Training_target = categorical(Training_target);
Target_testing = categorical(Target_testing);

target_total_train = dummyvar(Training_target);
target_total_test = dummyvar(Target_testing);
% target_total_train = Training_target;
% target_total_test = Target_testing;
disp("Total Testing size: " + string(size(target_total_test)));
disp("Total Training size: " + string(size(target_total_train)));
target_total_train = categorical(target_total_train);
target_total_test = categorical(target_total_test);
% Define the input shape
input_shape = [1 200];

% Define the number of output classes
num_classes = 3;

% Define the layers of the neural network
layers = [
    featureInputLayer(200)
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(25)
    reluLayer
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer
];

% Define the training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'Verbose', true, ...
    'ValidationData', {Combined_testing, Target_testing}, ...
    'ValidationFrequency', 10 );
   


% Train the neural network
[net, trainInfo] = trainNetwork(Combined_training, Training_target, layers, options);

% Plot the training and testing accuracy against epochs
figure;
plot(trainInfo.TrainingAccuracy);
hold on;
plot(trainInfo.ValidationAccuracy);
title('Accuracy vs Epoch');
xlabel('Epoch');
ylabel('Accuracy');
legend('Training Accuracy', 'Testing Accuracy');
