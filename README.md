# Cyber_Attack_Detection

Here is an implementation of cyber attack detection baased on MLP NN architecture.
The following steps are done and the code is offered in both matlab and python.
Step 1: Generate extensive closed-loop simulation data of the CSTR under the controller for various initial state conditions within a certain operating region; each simulation runs for a sufficient long period of time aiming to stabilize the CSTR at its steady-state. 
Step 2: Collect closed-loop data for the CSTR with and without min-max cyber-attack. 
Step 3: Attacked dataset is labeled as 1, and the dataset without cyber-attack is labeled as 0. An NN-based detector will be built to classify 0-1 and under noisy conditions(2). The input to the neural network is the dynamic trajectories of two states (CA and T).

## Data ##

1. CSTR_train_normal.mat: training input+output data under normal operation
CSTR_train_attack.mat: training input+output data under attack
CSTR_train_noise.mat: training input+output data under noisy measurement

2. CSTR_test_normal.mat: test input+output data under normal operation
CSTR_test_attack.mat: test input+output data under attack
CSTR_test_noise.mat: test input+output data under noisy measurement

Note: each data file (.mat) is a data matrix, where the number of rows represents the number of data samples, and the number of columns represents the number of time steps within each data sequence + last column is the output label: 0-no attack, 1-attack)
(For example, CSTR_train_normal.mat contains 694*201 data points -- there're 694 data sequences, where each data sequence includes 200 time steps (or called elements), and the last column = 0 represents no attack)

## Model ##

A Neural network model architecture implemented using TensorFlow's Keras API. It is consisited of four layers, including the input and output layers, and three hidden layers. The number of neurons in each hidden layer is 100, 50, and 25, respectively. Output layer is a Dense layer with 3 neurons, which corresponds to the number of classes the model is trying to predict. This layer uses the softmax activation function, which is commonly used for multi-class classification problems.

## Result ##

![image](https://user-images.githubusercontent.com/128442592/236099734-dc080c86-aaa1-4b31-a842-365e57043d8f.png)





