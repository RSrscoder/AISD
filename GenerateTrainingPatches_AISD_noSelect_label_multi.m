
%%% Generate the training data.
% 对分块后的数据进行筛选，剔除完全无阴影的块以及阴影像素个数较少的块

clear;close all;

addpath('utils');

batchSize      = 10;        %%% batch size
dataName      = 'TrainingPatches';
folder_shadow       = '..\datasets\AISD-Train412\shadow';
folder_mask         = '..\datasets\AISD-Train412\mask';


patchsize     = 256;
stride        = 64;
step          = 0;

count   = 0;

ext               =  {'*.jpg','*.png','*.bmp','*.jpeg','*.tif'};
filepaths_shadow           =  [];
filepaths_mask           =  [];

for i = 1 : length(ext)
    filepaths_shadow = cat(1,filepaths_shadow, dir(fullfile(folder_shadow, ext{i})));
    filepaths_mask = cat(1,filepaths_mask, dir(fullfile(folder_mask, ext{i})));
end

%% count the number of extracted patches
% scales  = [1 0.9 0.8 0.7];
scales  = [1 0.8 0.6];
ImageNums = length(filepaths_mask);
for i = 1 : ImageNums
    
    image_m = double(imread(fullfile(folder_mask,filepaths_mask(i).name)))/255;
    image_m = image_m(:,:,1);
    
    if mod(i,100)==0
        disp([i,length(filepaths_mask)]);
    end
    
    for s = 1:1
        image_m_scale = imresize(image_m,scales(s),'bicubic');
        [hei,wid,~] = size(image_m_scale);
        for x = 1+step : stride : (hei-patchsize+1)
            for y = 1+step :stride : (wid-patchsize+1)
                mask_patch   = image_m_scale(x : x+patchsize-1, y : y+patchsize-1,:);
                
                for j = 1:1
                    mask_aug = data_augmentation(mask_patch, j);
                    count = count+1;
                    %                     % 判断掩膜块中是否有阴影区域，且阴影像素个数大于50；如满足条件则加以统计
                    %                     if(sum(mask_aug(:))>50)
                    %                         count = count+1;
                    %                     end
                    
                end
                
            end
        end
    end
end

numPatches = ceil(count/batchSize)*batchSize;

disp('TotalPatchNumbers   batchSize  Iterations');
disp([numPatches,batchSize,numPatches/batchSize]);

%pause;

%% extract patches
inputs_shadow  = zeros(patchsize, patchsize, 3, numPatches,'single'); % this is fast
% inputs_ratio = zeros(patchsize, patchsize, 1, numPatches,'single'); % this is fast
% inputs_ratio2 = zeros(patchsize/2, patchsize/2, 1, numPatches,'single'); % this is fast
% inputs_ratio4 = zeros(patchsize/4, patchsize/4, 1, numPatches,'single'); % this is fast
% inputs_ratio8 = zeros(patchsize/8, patchsize/8, 1, numPatches,'single'); % this is fast
inputs_label  = zeros(patchsize, patchsize, 1, numPatches,'single'); % this is fast
inputs_label2 = zeros(patchsize/2, patchsize/2, 1, numPatches,'single'); % this is fast
inputs_label4 = zeros(patchsize/4, patchsize/4, 1, numPatches,'single'); % this is fast
inputs_label8 = zeros(patchsize/8, patchsize/8, 1, numPatches,'single'); % this is fast


count   = 0;
tic;
for i = 1 : ImageNums
    
    image_s = double(imread(fullfile(folder_shadow,filepaths_shadow(i).name)))/255; %
    image_m = double(imread(fullfile(folder_mask,filepaths_mask(i).name))); %
    image_m = image_m(:,:,1);
    image_m(image_m>0)=1;
    image_m(image_m<0)=0;   
    image_m = image_m+1;
    
%     %Compute (H +1)/(I +1) ratio images
%     hsi = rgb2hsi(image_s);
%     hsi_h = hsi(:,:,1);
%     hsi_s = hsi(:,:,2);
%     hsi_i = hsi(:,:,3);
%     ratio = (hsi_h + 1)./(hsi_i + 1);
%     
%     %Normalized (H +1)/(I +1) ratio images
%     min1 = min(ratio(:));
%     max1 = max(ratio(:));
%     ratio = (ratio - min1)/(max1 - min1);
    
    if mod(i,100)==0
        disp([i,length(filepaths_shadow)]);
    end
    
    for s = 1:1
        image_s_scale = im2single(imresize(image_s,scales(s),'bicubic'));
        image_m_scale = im2single(imresize(image_m,scales(s),'nearest'));
%         image_r_scale = im2single(imresize(ratio,scales(s),'bicubic'));
        
        [hei,wid,bands] = size(image_s_scale);
        %
        %         % 合并比例图到输入影像中
        %         image_s_scale(:,:,bands+1) = image_r_scale;
        
        for x = 1+step : stride : (hei-patchsize+1)
            for y = 1+step :stride : (wid-patchsize+1)
                image_patch   = image_s_scale(x : x+patchsize-1, y : y+patchsize-1,:);
%                 ratio_patch   = image_r_scale(x : x+patchsize-1, y : y+patchsize-1,:);
                mask_patch    = image_m_scale(x : x+patchsize-1, y : y+patchsize-1,:);
                
                for j = 1:1
                    image_aug = data_augmentation(image_patch, j);  % augment data
%                     ratio_aug = data_augmentation(ratio_patch, j);  % augment data
                    mask_aug  = data_augmentation(mask_patch, j);   % augment data
                    
                    count = count+1;
                    inputs_shadow(:, :, :, count) = image_aug;
%                     % Multi-ratio
%                     inputs_ratio(:, :, :, count)  = ratio_aug;
%                     inputs_ratio2(:, :, :, count)  = imresize(ratio_aug, 0.5);
%                     inputs_ratio4(:, :, :, count)  = imresize(ratio_aug, 0.25);
%                     inputs_ratio8(:, :, :, count)  = imresize(ratio_aug, 0.125);
                    % Multi-label
                    inputs_label(:, :, :, count)  = mask_aug;
                    inputs_label2(:, :, :, count)  = imresize(mask_aug, 0.5, 'nearest');
                    inputs_label4(:, :, :, count)  = imresize(mask_aug, 0.25, 'nearest');
                    inputs_label8(:, :, :, count)  = imresize(mask_aug, 0.125, 'nearest');  
                    
                    
                    %                     % 判断掩膜块中是否有阴影区域，且阴影像素个数大于50；如满足条件则加以统计
                    %                     if(sum(mask_aug(:))>50)
                    %                         count = count+1;
                    %                         inputs_shadow(:, :, :, count) = image_aug;
                    %                         inputs_label(:, :, :, count) = mask_aug;
                    %                     end
                    
                end
            end
            
        end
    end
end

toc;
set    = uint8(ones(1,size(inputs_shadow,4)));

disp('-------Datasize-------')
disp([size(inputs_shadow,4),batchSize,size(inputs_shadow,4)/batchSize]);

if ~exist(dataName,'file')
    mkdir(dataName);
end

%% save data
tic;
%save(fullfile(dataName,['imdb_rgb_',num2str(patchsize),'P_',num2str(batchSize),'B_',num2str(ImageNums),'_aug_select']), 'inputs_shadow','inputs_label','set','-v7.3');
% save(fullfile(dataName,['imdb_rgb_',num2str(patchsize),'P_AISD',num2str(ImageNums),'_noSelect_ratio_label_multi']), ...
%     'inputs_shadow','inputs_ratio','inputs_ratio2','inputs_ratio4','inputs_ratio8',...
%     'inputs_label','inputs_label2','inputs_label4','inputs_label8', 'set','-v7.3');
save(fullfile(dataName,['imdb_rgb_',num2str(patchsize),'P_AISD',num2str(ImageNums),'_noSelect_label_multi']), ...
    'inputs_shadow', 'inputs_label','inputs_label2','inputs_label4','inputs_label8', 'set','-v7.3');

toc;
