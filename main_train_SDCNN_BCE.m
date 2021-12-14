%   Main train_SDCNN
% -------------------------------------------------------------------------
%   Description:
%       Script to train SDCNN from scratch
%
%   Citation:
%       Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------


clc;close all; clear all;

% %% Basic parameters
% scale=3;    % upsampling scale
% depth=2;    % numbers of conv layers in each pyramid level
% gpu=1;      % GPU ID, 0 for CPU mode

%% setup MatConvnet
path_to_matconvnet = 'D:\DLwithMatlab\matconvnet';
fprintf('path to matconvnet library: %s\n', path_to_matconvnet);
run(fullfile(path_to_matconvnet, 'matlab\vl_setupnn.m'));

%% initialize opts
opts = init_SDCNN_opts;

%% save opts
filename = fullfile(opts.train.expDir, 'opts.mat');
fprintf('Save parameter %s\n', filename);
save(filename, 'opts');

%% setup paths
addpath(genpath('utils'));
%     addpath(fullfile(pwd, 'matconvnet/matlab'));
%     vl_setupnn;

%% initialize network
fprintf('Initialize network...\n');
model_filename = fullfile(opts.train.expDir, 'net-epoch-0.mat');

model = init_SDCNN_model(opts);
if( ~exist(model_filename, 'file') )
    fprintf('Save %s\n', model_filename);
    net = model.saveobj();
    save(model_filename, 'net');
else
    fprintf('Load %s\n', model_filename);
    model = load(model_filename);
    model = dagnn.DagNN.loadobj(model.net);
end

%% load imdb
% imdb_filename = fullfile('TrainingPatches', sprintf('imdb_rgb_%dP_UCF_OIRDS64_aug_noSelect.mat', opts.patch_size));
%imdb_filename = fullfile('TrainingPatches', sprintf('imdb_rgb_%dP_UCF_OIRDS64_aug_noSelect_Adjust.mat', opts.patch_size));
imdb_filename = fullfile('TrainingPatches', sprintf('imdb_rgb_%dP_AISD412_noSelect_label_multi.mat', opts.patch_size));

fprintf('Load data %s\n', imdb_filename);
imdb = load(imdb_filename);

%% training
[net, info] = vllab_cnn_train_dagNN_multi_label(model, imdb, opts.train);
