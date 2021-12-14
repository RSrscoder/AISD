% -------------------------------------------------------------------------
%   Description:
%       Script to evaluate DsCNN on benchmark datasets
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
clc;close all;clear all;

addpath('utils');
%% testing options
%model_scale = 4;            % model upsampling scale
%depth = 2;
gpu         = 0;            % GPU ID, gpu = 0 for CPU mode
epochNum = 100;             % the epoch number needed to be test
% dataset     = 'AISD-Test51';
dataset     = 'HighRes-Test';
% dataset     = 'WHU-RSSD-Train450';


% dataset     = 'UCF-OIRDS74';

% dataset     = 'UCF-OIRDS-Test10';
%dataset     = 'UCF-OIRDS-Train64';
%dataset     = 'Test20_ISTD_Adjust';
%dataset     = 'Test50_ISTD_Adjust';
% dataset     = 'UCF-355';
%dataset     = 'UIUC-76';
%dataset     = 'Test540_ISTD_Adjust';
%dataset     = 'Train100_ISTD_Adjust';
compute_ifc = 0;            % IFC calculation is slow, enable when needed

% tic;
%% load model
% initialize opts
opts = init_SDCNN_opts;

% setup model name
for i = 1:length(opts.train_dataset)
    opts.data_name = sprintf('%s',opts.train_dataset{i});
end

opts.net_name = sprintf('SDCNN_%s', opts.loss);

% opts.model_name = sprintf('%s_%s_pw%d_lr%s_step%d_drop%s_min%s_bs%d_depth_%d_level1_relu_improve_noselect_res_BN_U_BRC_crop_fuse_Adjust_step_pool', ...
%     opts.net_name, ...
%     opts.data_name, opts.patch_size, ...
%     num2str(opts.lr), opts.lr_step, ...
%     num2str(opts.lr_drop), num2str(opts.lr_min), ...
%     opts.batch_size,opts.depth);

opts.model_name = sprintf('%s_%s', ...
    opts.net_name, ...
    opts.data_name);

% setup paths
input_dir = fullfile('datasets', dataset);
output_dir = fullfile('results', dataset, sprintf('%s', opts.model_name));


avg_Total_epoch_deconv = zeros(epochNum,1);
avg_F_epoch_deconv = zeros(epochNum,1);
avg_BER_epoch_deconv = zeros(epochNum,1);

%% 计算每个epoch训练得到模型的测试精度
for k = 100:epochNum
    model_filename = fullfile('models', sprintf('%s', opts.model_name),sprintf('net-epoch-%d.mat', k));    
    fprintf('Load %s\n', model_filename);
    
    net = load(model_filename);
    net = dagnn.DagNN.loadobj(net.net);
    net.addLayer('Softmax', dagnn.SoftMax(), {'level4_prediction'}, {'level4_prediction_softmax'}); 
    net.mode = 'test' ;
    
%     output_var = 'level4_prediction';
%     output_index = net.getVarIndex(output_var);
%     net.vars(output_index).precious = 1;                               
    
    if( gpu ~= 0 )
        gpuDevice(gpu)
        net.move('gpu');
    end
    
    %% setup paths
    output_dir_epoch = fullfile('results', dataset,  ...
        sprintf('%s', opts.model_name),sprintf('%s', 'epoch',num2str(k)));
    output_BW_dir_epoch = fullfile('results', dataset,  ...
        sprintf('%s', opts.model_name),sprintf('%s', 'epoch',num2str(k)),'BW');
    
    
    if( ~exist(output_dir_epoch, 'dir') )
        mkdir(output_dir_epoch);
    end
    
    if( ~exist(output_BW_dir_epoch, 'dir') )
        mkdir(output_BW_dir_epoch);
    end
    
    addpath(genpath('utils'));
    
    %% load image
    % WHU-rssd
    folderTest_s  = fullfile('..\datasets',dataset,'shadow');
    % folderTest_m  = fullfile('datasets',dataset,'mask_Rename_cut');
%     folderTest_m  = fullfile('..\datasets',dataset,'mask');
    folderTest_m  = fullfile('..\datasets',dataset,'mask_Rename');
    
    % folderTest_s  = fullfile('datasets',dataset,'shadow_cut'); %%% test dataset
    % folderTest_m  = fullfile('datasets',dataset,'maskAdjust_cut'); %%% test dataset
    
    %%% read images
    ext         =  {'*.jpg','*.png','*.bmp','*.tif'};
    filePaths_s   =  [];
    filePaths_m   =  [];
    for i = 1 : length(ext)
        filePaths_s = cat(1,filePaths_s, dir(fullfile(folderTest_s,ext{i})));
        filePaths_m = cat(1,filePaths_m, dir(fullfile(folderTest_m,ext{i})));
    end
    num_img = length(filePaths_m);
    
    %% testing
    Accuracy = zeros(num_img,7);
    TP_stat =  zeros(num_img,1);     % true positive, the number of true shadow pixels which are identified correctly
    TN_stat =  zeros(num_img,1);     % true negative, the number of nonshadow pixels which are identified correctly
    FP_stat =  zeros(num_img,1);     % false positive, the number of nonshadow pixels which are identified as true shadow pixels
    FN_stat =  zeros(num_img,1);     % false negative, the number of true shadow pixels which are identified as nonshadow pixels
    count_stat = zeros(num_img,1);     % sum of the image pixels
    Mean_acc = zeros(1,7);  % mean value of accuracy
    Std_acc = zeros(1,7);   % std value of accuracy
    
    threM = zeros(num_img,1);
    Name = cell(num_img,1);
    tElapsed = zeros(num_img,1);
    
    for i = 1:num_img
        
        %% read images
        image_s_name = filePaths_s(i).name;
        image_s = double(imread(fullfile(folderTest_s,image_s_name)))/255;
        %img_name = img_list{i};
        fprintf('Testing SDCNN on %s: %d/%d: %s\n', dataset, i, num_img, image_s_name);
        
        %% Load GT image
        image_m_name = filePaths_m(i).name;
        image_GT = double(imread(fullfile(folderTest_m,image_m_name)))/255;
        image_GT = image_GT(:,:,1);
        image_GT(image_GT>0)=1;
        [m,n,channel] = size(image_GT);
        %
        %     mask = image_GT;
        %     mask() =
        
%         %% Compute (H +1)/(I +1) ratio images
%         hsi = rgb2hsi(image_s);
%         hsi_h = hsi(:,:,1);
%         hsi_s = hsi(:,:,2);
%         hsi_i = hsi(:,:,3);
%         ratio = (hsi_h + 1)./(hsi_i + 1);
%         
%         %Normalized (H +1)/(I +1) ratio images
%         min1 = min(ratio(:));
%         max1 = max(ratio(:));
%         ratio = (ratio - min1)/(max1 - min1);
%         
%         %% Resize ratio image
%         ratio2  = imresize(ratio, 0.5);
%         ratio4  = imresize(ratio, 0.25);
%         ratio8  = imresize(ratio, 0.125);                     
        

        tStart = tic; 
        %% apply DsCNN
        net.mode = 'test' ;
        output_var = 'level4_prediction_softmax';
        output_index = net.getVarIndex(output_var);
        net.vars(output_index).precious = 1;
                
        % convert to GPU
        y = single(image_s);
        
        if( gpu )
            y = gpuArray(y);
        end
        
        % forward
        tic;
        inputs = {'input', y};
        net.eval(inputs);
        time = toc;
        
        y = gather(net.vars(output_index).value);
        
        output = double(y);
        output_1 = output(:,:,1);
        output_2 = output(:,:,2); % shadow probability
        
        %         y = gather(net.vars(output_index).value);
        %         output = double(y);
        
        %         % 根据图像大小来决定是否使用GPU
        %         if m>1000
        %             GPU = 0;
        %         else
        %             GPU = 1;
        %         end
        
        %         output = exc_SDCNN(image_s, net, gpu);
        
        %% 二值化
        %     thre = 0.5;
        thre = graythresh(output_2);
        threM(i,1) = thre;
        output_BW = zeros(size(output_2));
        output_BW(output_2>thre)=1;
        output_BW(output_2<thre)=0;
        
        tElapsed(i,1) = toc(tStart);
        %% save result
        % 保存网络直接输出的概率图
        output_filename = fullfile(output_dir_epoch, sprintf('%s', image_s_name));
        %         fprintf('Save %s\n', output_filename);
        imwrite(output_2, output_filename);
        
        % 保存二值化结果图
        output_BW_filename = fullfile(output_BW_dir_epoch, sprintf('%s', image_s_name));
        imwrite(output_BW, output_BW_filename);
        
        %% evaluate
        Accuracy(i,:) = accuracy(output_BW, image_GT);
        [TP_stat(i,1),TN_stat(i,1),FP_stat(i,1),FN_stat(i,1),count_stat(i,1)] = accuracy_indiv(output_BW, image_GT);
        
        Name{i,1} = filePaths_s(i).name; % 精度矩阵中加入对应影像名称,便于识别
    end
    
    %% 统计平均
    Mean_acc = mean(Accuracy);
    Std_acc = std(Accuracy);
    Accuracy(end+1,:) = Mean_acc;
    Accuracy(end+1,:) = Std_acc;
    
    avg_pro_s = Accuracy(end-1,1);
    avg_pro_n = Accuracy(end-1,2);
    avg_user_s = Accuracy(end-1,3);
    avg_user_n = Accuracy(end-1,4);
    avg_Total = Accuracy(end-1,5);
    avg_F = Accuracy(end-1,6);
    avg_BER = Accuracy(end-1,7);
    
    % Show
    %     fprintf('Average producer accuracy of shadow = %f\n', avg_pro_s);
    %     fprintf('Average producer accuracy of nonshadow = %f\n', avg_pro_n);
    %     fprintf('Average user accuracy of shadow = %f\n',avg_user_s);
    %     fprintf('Average user accuracy of nonshadow = %f\n', avg_user_n);
    fprintf('Average total accuracy = %f\n', avg_Total);
    fprintf('Average F score = %f\n', avg_F);
    fprintf('Average BER = %f\n', avg_BER);
    
    % 保存每个epoch的测试精度平均值
    avg_Total_epoch_deconv(k,1) = avg_Total;
    avg_F_epoch_deconv(k,1) = avg_F;
    avg_BER_epoch_deconv(k,1) = avg_BER;
    
    
    %% Save
    % 保存平均精度
    filename_acc_avg = fullfile(output_dir_epoch, 'Accuracy_avg.txt');
    save_matrix(Accuracy, filename_acc_avg);
    
    % 保存阈值
    filename_thre = fullfile(output_dir_epoch, 'Threshold.txt');
    save_matrix(threM, filename_thre);

    % 保存运行时间
    filename_time = fullfile(output_dir_epoch, 'Time.txt');
    save_matrix(tElapsed, filename_time);     
    
    %保存文件名
    %fid=fopen('Tappen_model\Name.txt','w');
    fid=fopen(fullfile(output_dir_epoch,'\','Name.txt'),'w');
    for i=1:size(Name,1)
        a = Name(i);
        a = cell2mat(a);
        fprintf(fid,'%s\n',a);
    end
    
end

% %% Draw Total Accuracy Curve
% figure;
% hold on;
% 
% % Draw the curve
% plot(avg_Total_epoch_deconv,'-r','DisplayName','Deconv and concat With Multi-Attention Step Fusion Structure');
% % plot(PSNRs_mean_noRes,'-b','DisplayName','Without Residual Learning');
% 
% % Add Title and Axis Labels
% % title('Total Accuracy');
% xlabel('Epochs');                 % 设置坐标轴标签
% ylabel('Total Accuracy');
% 
% grid;                           % 打开绘图网线
% 
% %set(gca,'ytick',[])           %删除 当前图 y 轴刻度
% % % axis([0 linelength 0 260]);     % 用来设置坐标轴显示的最大值最小值
% % % set(gca,'YTick',[0:50:260]);    % 对坐标轴的刻度的分度进行设置；
% 
% %加图例
% legend('show');
% 
% hold off;
% 
% %% Draw F-score Curve
% figure;
% hold on;
% 
% % Draw the curve
% plot(avg_F_epoch_deconv,'-r','DisplayName','Deconv and concat With Multi-Attention Step Fusion Structure');
% % plot(SSIMs_mean_noRes,'-b','DisplayName','Without Residual Learning');
% 
% % Add Title and Axis Labels
% %title('Profile');
% xlabel('Epochs');                 % 设置坐标轴标签
% ylabel('F-score');
% 
% grid;                           % 打开绘图网线
% 
% %set(gca,'ytick',[])           %删除 当前图 y 轴刻度
% 
% %加图例
% legend('show');
% 
% hold off;
% 
% %% Draw BER Curve
% figure;
% hold on;
% 
% % Draw the curve
% plot(avg_BER_epoch_deconv,'-r','DisplayName','Deconv and concat With Multi-Attention Step Fusion Structure');
% % plot(SSIMs_mean_noRes,'-b','DisplayName','Without Residual Learning');
% 
% % Add Title and Axis Labels
% %title('Profile');
% xlabel('Epochs');                 % 设置坐标轴标签
% ylabel('BER');
% 
% grid;                           % 打开绘图网线
% 
% %set(gca,'ytick',[])           %删除 当前图 y 轴刻度
% 
% %加图例
% legend('show');
% 
% hold off;
% 
% 
% %% Save
% % Overall accuracy
% filename_Total_avg_epoch = fullfile(output_dir, 'Total_avg_epoch_deconv_con_attentionD.txt');
% save_matrix(avg_Total_epoch_deconv, filename_Total_avg_epoch);
% save([output_dir,'\','Total_avg_epoch_deconv_con_attentionD.mat'],'avg_Total_epoch_deconv','-v7.3');
% 
% % F-score accuracy
% filename_F_avg_epoch = fullfile(output_dir, 'Fscore_avg_epoch_deconv_con_attentionD.txt');
% save_matrix(avg_F_epoch_deconv, filename_F_avg_epoch);
% save([output_dir,'\','Fscore_avg_epoch_deconv_con_attentionD.mat'],'avg_F_epoch_deconv','-v7.3');
% 
% % Balance Error Rate (BER)
% filename_BER_avg_epoch = fullfile(output_dir, 'BER_avg_epoch_deconv_con_attentionD.txt');
% save_matrix(avg_BER_epoch_deconv, filename_BER_avg_epoch);
% save([output_dir,'\','BER_avg_epoch_deconv_con_attentionD.mat'],'avg_BER_epoch_deconv','-v7.3');
% 
% 
% toc;
