function opts = init_SDCNN_opts()
% -------------------------------------------------------------------------
%   Description:
%       Generate model options for SDCNN
%
%   Input:
%       - scale : upsampling scale
%       - depth : number of conv layers in one pyramid level
%       - gpu   : GPU ID, 0 for CPU mode
%
%   Output:
%       - opts  : options for DsCNN
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

%% network options
% opts.scale              = 3;
opts.depth              = 4;    % 网络尺度数
% opts.lc                 = 1;    % 每个尺度层上的卷积数
opts.weight_decay       = 0.0001;
opts.init_sigma         = 0.001;
opts.conv_f             = 3;
opts.conv_n             = 64;
%opts.loss               = 'L1';
% opts.loss               = 'L2';
opts.loss               = 'softmaxlog';

%% training options
opts.gpu                = 1;
%opts.batch_size         = 64;
%opts.batch_size         = 32;
opts.batch_size         = 10;
opts.num_train_batch    = 100;     % number of training batch in one epoch
opts.num_valid_batch    = 100;      % number of validation batch in one epoch
% opts.lr                 = 1e-3;     % initial learning rate
opts.lr                 = 1e-4;     % initial learning rate
opts.lr_step            = 20;       % number of epochs to drop learning rate
opts.lr_drop            = 0.5;      % learning rate drop ratio
opts.lr_min             = 1e-6;     % minimum learning rate
opts.patch_size         = 256;
%opts.patch_size         = 256;
opts.data_augmentation  = 1;
opts.scale_augmentation = 1;
% opts.gclip = 1;

%% dataset options
opts.train_dataset          = {};
%opts.train_dataset{end+1}   = 'Train100_ISTD_Adjust';
%opts.train_dataset{end+1}   = 'UCF-OIRDS-Train64';
opts.train_dataset{end+1}   = 'AISD-Train412';
% opts.valid_dataset          = {};
% opts.valid_dataset{end+1}   = 'Test20_ISTD_Adjust';


%% setup model name
% opts.data_name = 'train';
% for i = 1:length(opts.train_dataset)
%     opts.data_name = sprintf('%s_%s', opts.data_name, opts.train_dataset{i});
% end
for i = 1:length(opts.train_dataset)
    opts.data_name = sprintf('%s',opts.train_dataset{i});
end

opts.net_name = sprintf('SDCNN_%s', opts.loss);

% opts.model_name = sprintf('%s_%s_pw%d_lr%s_step%d_drop%s_min%s_bs%d_depth_%d_level1_relu_improve_noselect_res_BN_U_BRC_crop_fuse', ...
%     opts.net_name, ...
%     opts.data_name, opts.patch_size, ...
%     num2str(opts.lr), opts.lr_step, ...
%     num2str(opts.lr_drop), num2str(opts.lr_min), ...
%     opts.batch_size,opts.depth);

opts.model_name = sprintf('%s_%s', ...
    opts.net_name, ...
    opts.data_name);


%% setup dagnn training parameters
if( opts.gpu == 0 )
    opts.train.gpus     = [];
else
    opts.train.gpus     = [opts.gpu];
end
opts.train.batchSize    = opts.batch_size;
% opts.train.numEpochs    = 1000;
opts.train.numEpochs    = 100;
opts.train.continue     = true;
opts.train.learningRate = learning_rate_policy(opts.lr, opts.lr_step, opts.lr_drop, ...
    opts.lr_min, opts.train.numEpochs);

opts.train.expDir = fullfile('models', opts.model_name) ; % model output dir
if( ~exist(opts.train.expDir, 'dir') )
    mkdir(opts.train.expDir);
end

opts.train.model_name       = opts.model_name;
opts.train.num_train_batch  = opts.num_train_batch;
opts.train.num_valid_batch  = opts.num_valid_batch;

% setup loss
% opts.train.derOutputs = {'loss',1};
% opts.level = ceil(log(opts.scale) / log(2));
opts.level = opts.depth;
opts.train.derOutputs = {};
for s = opts.level : -1 : 1
    opts.train.derOutputs{end+1} = sprintf('level%d_BCE_loss', s);
    opts.train.derOutputs{end+1} = 1;
end


end