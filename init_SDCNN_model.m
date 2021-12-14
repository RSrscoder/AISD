function net = init_SDCNN_model(opts)
% -------------------------------------------------------------------------
%   Description:
%       initialize SDCNN model, 先激活后卷积
%
%   Input:
%       - opts  : options generated from init_SDCNN_opts()
%
%   Output:
%       - net   : dagnn model
%
% -------------------------------------------------------------------------

%% parameters
rng('default');
rng(0) ;

f       = opts.conv_f;
fd      = 4;
n       = opts.conv_n;
pad     = floor(f/2);
%sigma   = opts.init_sigma;
depth   = opts.depth;
% lc = opts.lc;

if( f == 3 )
    crop = [0, 1, 0, 1];
elseif( f == 5 )
    crop = [1, 2, 1, 2];
else
    error('Need to specify crop in deconvolution for f = %d\n', f);
end

%% initialize model
net = dagnn.DagNN;

num = 1; lyn = 1;
sum_layer_count =1;
crop_count = 1;
sigma = sqrt(2/(f*f*n));

% First conv layer
convBlock = dagnn.Conv('size', [f,f,3,n], 'pad', pad, 'stride', 1, 'hasBias', true);
net.addLayer(['conv', num2str(lyn)], convBlock, {'input'}, {['x', num2str(num)]}, {['conv', num2str(lyn), '_f'], ['conv', num2str(lyn), '_b']});
initializeLearningRate(net, ['conv',num2str(lyn)],sigma,f,3,n);
% BN
net.addLayer(['bn', num2str(lyn)], dagnn.BatchNorm('numChannels', n), {['x', num2str(num)]}, {['x', num2str(num+1)]}, {['bn', num2str(lyn), '_f'], ['bn', num2str(lyn), '_b'], ['bn', num2str(lyn), '_m']});
initializeLearningRateBN(net, ['bn', num2str(lyn)],sigma,n);
% ReLU
net.addLayer(['relu', num2str(lyn)], dagnn.ReLU(), {['x', num2str(num+1)]}, {['x', num2str(num+2)]});

% Second conv layer
convBlock = dagnn.Conv('size', [f,f,n,n], 'pad', pad, 'stride', 1, 'hasBias', true);
net.addLayer(['conv', num2str(lyn+1)], convBlock, {['x', num2str(num+2)]}, {['x', num2str(num+3)]}, {['conv', num2str(lyn+1), '_f'], ['conv', num2str(lyn+1), '_b']});
initializeLearningRate(net, ['conv',num2str(lyn+1)],sigma,f,n,n);
% BN
net.addLayer(['bn', num2str(lyn+1)], dagnn.BatchNorm('numChannels', n), {['x', num2str(num+3)]}, {['x', num2str(num+4)]}, {['bn', num2str(lyn+1), '_f'], ['bn', num2str(lyn+1), '_b'], ['bn', num2str(lyn+1), '_m']});
initializeLearningRateBN(net, ['bn', num2str(lyn+1)],sigma,n);
% ReLU
net.addLayer(['relu', num2str(lyn+1)], dagnn.ReLU(), {['x', num2str(num+4)]}, {['x', num2str(num+5)]});

% Third conv layer
convBlock = dagnn.Conv('size', [f,f,n,n], 'pad', pad, 'stride', 1, 'hasBias', true);
net.addLayer(['conv', num2str(lyn+2)], convBlock, {['x', num2str(num+5)]}, {['x', num2str(num+6)]}, {['conv', num2str(lyn+2), '_f'], ['conv', num2str(lyn+2), '_b']});
initializeLearningRate(net, ['conv',num2str(lyn+2)],sigma,f,n,n);
% BN
net.addLayer(['bn', num2str(lyn+2)], dagnn.BatchNorm('numChannels', n), {['x', num2str(num+6)]}, {['x', num2str(num+7)]}, {['bn', num2str(lyn+2), '_f'], ['bn', num2str(lyn+2), '_b'], ['bn', num2str(lyn+2), '_m']});
initializeLearningRateBN(net, ['bn', num2str(lyn+2)],sigma,n);
% ReLU
net.addLayer(['relu', num2str(lyn+2)], dagnn.ReLU(), {['x', num2str(num+7)]}, {['x', num2str(num+8)]});

% sum
net.addLayer(['cropLayer', num2str(crop_count)], dagnn.Crop(), {['x', num2str(num+2)], ['x', num2str(num+8)]}, {['x', num2str(num+9)]});
net.addLayer(['sumLayer', num2str(sum_layer_count)], dagnn.Sum(), {['x', num2str(num+9)], ['x', num2str(num+8)]}, ['x', num2str(num+10)]);

num = num+10;
lyn = lyn+2;
sum_layer_count = sum_layer_count +1;

%% Encoder
fn = n;  % 特征通道数
%pl = 1;
pool_count=1;
sconv_count = 1;
res = {''}; res_count = 1;
%sum_layer_count =1;
for d = 2:depth
    fn = 2*fn;
    
    res{res_count} = ['x', num2str(num)];
    res_count = res_count+1;
    
    %     % stride conv
    %     SconvBlock = dagnn.Conv('size', [f,f,fn/2,fn], 'pad', pad, 'stride', 2, 'hasBias', true);
    %     net.addLayer(['Sconv', num2str(sconv_count)], SconvBlock, {['x', num2str(num)]}, {['x', num2str(num+1)]}, {['Sconv', num2str(sconv_count), '_f'], ['Sconv', num2str(sconv_count), '_b']});
    %     initializeLearningRate(net, ['Sconv',num2str(sconv_count)],sigma,f,fn/2,fn);
    
    % Pooling
    net.addLayer(['Pool', num2str(pool_count)], dagnn.Pooling('poolSize', [2 2], 'stride', 2), {['x', num2str(num)]}, {['x', num2str(num+1)]});
    
    % Conv
    convBlock = dagnn.Conv('size', [f,f,fn/2,fn], 'pad', pad, 'stride', 1, 'hasBias', true);
    net.addLayer(['conv', num2str(lyn+1)], convBlock, {['x', num2str(num+1)]}, {['x', num2str(num+2)]}, {['conv', num2str(lyn+1), '_f'], ['conv', num2str(lyn+1), '_b']});
    initializeLearningRate(net, ['conv',num2str(lyn+1)],sigma,f,fn/2,fn);
    % BN
    net.addLayer(['bn', num2str(lyn+1)], dagnn.BatchNorm('numChannels', fn), {['x', num2str(num+2)]}, {['x', num2str(num+3)]}, {['bn', num2str(lyn+1), '_f'], ['bn', num2str(lyn+1), '_b'], ['bn', num2str(lyn+1), '_m']});
    initializeLearningRateBN(net, ['bn', num2str(lyn+1)],sigma,fn);
    % ReLU
    net.addLayer(['relu', num2str(lyn+1)], dagnn.ReLU(), {['x', num2str(num+3)]}, {['x', num2str(num+4)]});
    
    % conv
    convBlock = dagnn.Conv('size', [f,f,fn,fn], 'pad', pad, 'stride', 1, 'hasBias', true);
    net.addLayer(['conv', num2str(lyn+2)], convBlock, {['x', num2str(num+4)]}, {['x', num2str(num+5)]}, {['conv', num2str(lyn+2), '_f'], ['conv', num2str(lyn+2), '_b']});
    initializeLearningRate(net, ['conv',num2str(lyn+2)],sigma,f,fn,fn);
    % BN
    net.addLayer(['bn', num2str(lyn+2)], dagnn.BatchNorm('numChannels', fn), {['x', num2str(num+5)]}, {['x', num2str(num+6)]}, {['bn', num2str(lyn+2), '_f'], ['bn', num2str(lyn+2), '_b'], ['bn', num2str(lyn+2), '_m']});
    initializeLearningRateBN(net, ['bn', num2str(lyn+2)],sigma,fn);
    % ReLU
    net.addLayer(['relu', num2str(lyn+2)], dagnn.ReLU(), {['x', num2str(num+6)]}, {['x', num2str(num+7)]});
    
    % conv
    convBlock = dagnn.Conv('size', [f,f,fn,fn], 'pad', pad, 'stride', 1, 'hasBias', true);
    net.addLayer(['conv', num2str(lyn+3)], convBlock, {['x', num2str(num+7)]}, {['x', num2str(num+8)]}, {['conv', num2str(lyn+3), '_f'], ['conv', num2str(lyn+3), '_b']});
    initializeLearningRate(net, ['conv',num2str(lyn+3)],sigma,f,fn,fn);
    % BN
    net.addLayer(['bn', num2str(lyn+3)], dagnn.BatchNorm('numChannels', fn), {['x', num2str(num+8)]}, {['x', num2str(num+9)]}, {['bn', num2str(lyn+3), '_f'], ['bn', num2str(lyn+3), '_b'], ['bn', num2str(lyn+3), '_m']});
    initializeLearningRateBN(net, ['bn', num2str(lyn+3)],sigma,fn);
    % ReLU
    net.addLayer(['relu', num2str(lyn+3)], dagnn.ReLU(), {['x', num2str(num+9)]}, {['x', num2str(num+10)]});
    
    % sum
    net.addLayer(['cropLayer', num2str(crop_count+1)], dagnn.Crop(), {['x', num2str(num+4)], ['x', num2str(num+10)]}, {['x', num2str(num+11)]});
    net.addLayer(['sumLayer', num2str(sum_layer_count)], dagnn.Sum(), {['x', num2str(num+11)], ['x', num2str(num+10)]}, ['x', num2str(num+12)]);
    
    num = num+12;
    lyn = lyn+3;
    pool_count = pool_count + 1;
    %pl = pl+1;
    sum_layer_count = sum_layer_count+1;
    sconv_count = sconv_count + 1;
    crop_count = crop_count + 1;
end

% 保存第一个待融合尺度
scale_count = 1;
scale{scale_count} = ['x', num2str(num)];
upscale(scale_count) = depth -1;
channel(scale_count) = fn;
scale_count = scale_count+1;


%% Decoder
ct = 1;
% unpool_count = 1;
% sum_count = 1;
cropSkip_count = 1;
concatSkip_count = 1;

for d = depth:-1:2
    fn = fn/2;
    
    
    % ConvTranspose
    convtBlock=dagnn.ConvTranspose('size', [f,f,fn,fn*2], 'upsample', 2, 'crop', crop, 'hasBias', true);
    %     convtBlock=dagnn.ConvTranspose('size', [fd,fd,fn,fn*2], 'upsample', 2, 'crop', 1, 'hasBias', true);
    net.addLayer(['convt', num2str(ct)], convtBlock, {['x', num2str(num)]}, {['x', num2str(num+1)]}, {['convt', num2str(ct), '_f'], ['convt', num2str(ct), '_b']});
    %net.addLayer(['convt', num2str(ct)], convtBlock(f,f,n,n), {['x', num2str(num)]}, {['x', num2str(num+1)]}, {['convt', num2str(ct), '_f'], ['convt', num2str(ct), '_b']});
    initializeLearningRateConvT(net, ['convt', num2str(ct)],sigma,f,fn,fn*2);
    
    %     % Unpooling
    %     net.addLayer(['unpool', num2str(unpool_count)], dagnn.Unpooling(), {['x', num2str(num)]}, {['x', num2str(num+1)]});
    %     % Conv
    %     convBlock = dagnn.Conv('size', [f,f,fn*2,fn], 'pad', pad, 'stride', 1, 'hasBias', true);
    %     net.addLayer(['conv', num2str(lyn+1)], convBlock, {['x', num2str(num+1)]}, {['x', num2str(num+2)]}, {['conv', num2str(lyn+1), '_f'], ['conv', num2str(lyn+1), '_b']});
    %     initializeLearningRate(net, ['conv',num2str(lyn+1)],sigma,f,fn*2,fn);
    
    %     % sum
    %     res_count = res_count-1;
    %     net.addLayer(['cropSkip', num2str(cropSkip_count)], dagnn.Crop(), { ['x', num2str(num+1)],res{res_count}}, {['x', num2str(num+2)]});
    %     net.addLayer(['sumSkip', num2str(sum_count)], dagnn.Sum(), {['x', num2str(num+2)], res{res_count}}, ['x', num2str(num+3)]);
    
    % Concat
    res_count = res_count-1;
    net.addLayer(['cropSkipCat', num2str(cropSkip_count)], dagnn.Crop(), { ['x', num2str(num+1)],res{res_count}}, {['x', num2str(num+2)]});
    net.addLayer(['concatSkip', num2str(concatSkip_count)], dagnn.Concat('dim', 3), {['x', num2str(num+2)], res{res_count}}, {['x', num2str(num+3)]});
    
    % conv
    convBlock = dagnn.Conv('size', [f,f,2*fn,fn], 'pad', pad, 'stride', 1, 'hasBias', true);
    net.addLayer(['conv', num2str(lyn+1)], convBlock, {['x', num2str(num+3)]}, {['x', num2str(num+4)]}, {['conv', num2str(lyn+1), '_f'], ['conv', num2str(lyn+1), '_b']});
    initializeLearningRate(net, ['conv',num2str(lyn+1)],sigma,f,2*fn,fn);
    % BN
    net.addLayer(['bn', num2str(lyn+1)], dagnn.BatchNorm('numChannels', fn), {['x', num2str(num+4)]}, {['x', num2str(num+5)]}, {['bn', num2str(lyn+1), '_f'], ['bn', num2str(lyn+1), '_b'], ['bn', num2str(lyn+1), '_m']});
    initializeLearningRateBN(net, ['bn', num2str(lyn+1)],sigma,fn);
    % ReLU
    net.addLayer(['relu', num2str(lyn+1)], dagnn.ReLU(), {['x', num2str(num+5)]}, {['x', num2str(num+6)]});
    
    
    % conv
    convBlock = dagnn.Conv('size', [f,f,fn,fn], 'pad', pad, 'stride', 1, 'hasBias', true);
    net.addLayer(['conv', num2str(lyn+2)], convBlock, {['x', num2str(num+6)]}, {['x', num2str(num+7)]}, {['conv', num2str(lyn+2), '_f'], ['conv', num2str(lyn+2), '_b']});
    initializeLearningRate(net, ['conv',num2str(lyn+2)],sigma,f,fn,fn);
    % BN
    net.addLayer(['bn', num2str(lyn+2)], dagnn.BatchNorm('numChannels', fn), {['x', num2str(num+7)]}, {['x', num2str(num+8)]}, {['bn', num2str(lyn+2), '_f'], ['bn', num2str(lyn+2), '_b'], ['bn', num2str(lyn+2), '_m']});
    initializeLearningRateBN(net, ['bn', num2str(lyn+2)],sigma,fn);
    % ReLU
    net.addLayer(['relu', num2str(lyn+2)], dagnn.ReLU(), {['x', num2str(num+8)]}, {['x', num2str(num+9)]});
    
    % conv
    convBlock = dagnn.Conv('size', [f,f,fn,fn], 'pad', pad, 'stride', 1, 'hasBias', true);
    net.addLayer(['conv', num2str(lyn+3)], convBlock, {['x', num2str(num+9)]}, {['x', num2str(num+10)]}, {['conv', num2str(lyn+3), '_f'], ['conv', num2str(lyn+3), '_b']});
    initializeLearningRate(net, ['conv',num2str(lyn+3)],sigma,f,fn,fn);
    % BN
    net.addLayer(['bn', num2str(lyn+3)], dagnn.BatchNorm('numChannels', fn), {['x', num2str(num+10)]}, {['x', num2str(num+11)]}, {['bn', num2str(lyn+3), '_f'], ['bn', num2str(lyn+3), '_b'], ['bn', num2str(lyn+3), '_m']});
    initializeLearningRateBN(net, ['bn', num2str(lyn+3)],sigma,fn);
    % ReLU
    net.addLayer(['relu', num2str(lyn+3)], dagnn.ReLU(), {['x', num2str(num+11)]}, {['x', num2str(num+12)]});
    
    
    % sum
    net.addLayer(['cropLayer', num2str(crop_count+1)], dagnn.Crop(), {['x', num2str(num+6)], ['x', num2str(num+12)]}, {['x', num2str(num+13)]});
    net.addLayer(['sumLayer', num2str(sum_layer_count)], dagnn.Sum(), {['x', num2str(num+13)], ['x', num2str(num+12)]}, ['x', num2str(num+14)]);
    
    num = num+14;
    lyn = lyn+3;
    ct = ct+1;
    %     sum_count = sum_count+1;
    %     unpool_count = unpool_count+1;
    sum_layer_count = sum_layer_count+1;
    crop_count = crop_count+1;
    cropSkip_count = cropSkip_count+1;
    concatSkip_count = concatSkip_count+1;
    
    % 保存待融合尺度
    scale{scale_count} = ['x', num2str(num)];
    upscale(scale_count) = d -2;
    channel(scale_count) = fn;
    scale_count = scale_count+1;
    
end

%% 特征融合层
outScale = scale_count-1;
% fuse = {''};
% up_count = 1;
cropCat_count = 1;
concat_count = 1;
%poolAt_count = 1;
% dotP_count = 1;
ctFuse_count = 1;
convSD_count = 1;
loss_count = 1;

for s=1:outScale
    
    if s==1   % The first fuse block
        fn1 = channel(s);
        fn2 = channel(s+1);
        %         fn3 = channel(s+2);
        fuse_fn = 2*fn2;
        
        %         % Pooling
        %         net.addLayer(['Pool_Att', num2str(poolAt_count)], dagnn.Pooling('poolSize', [8 8], 'stride', 8), {'ratio'}, {['x', num2str(num+1)]});
        
%         % Ratio Attention
%         net.addLayer(['dot_prod_' num2str(dotP_count)], dagnn.DotProduct(), {scale{s}, 'ratio_down8'}, {['x', num2str(num+1)]}) ;
        
        % 1*1 conv
        %         prediction_name = { sprintf('level%d_prediction', convSD_count) };
        convBlock = dagnn.Conv('size', [1,1,fn1,2], 'pad', 0, 'stride', 1, 'hasBias', true);
        net.addLayer(['convSD', num2str(convSD_count)], convBlock, {scale{s}}, {['level', num2str(convSD_count),'_prediction']}, {['convSD', num2str(convSD_count), '_f'], ['convSD', num2str(convSD_count), '_b']});
        initializeLearningRate(net, ['convSD',num2str(convSD_count)],sigma,1,fn1,2);
        
        % loss layer
        %         loss_name = { sprintf('level%d_%s_loss', loss_count, opts.loss) };
        %         loss_input = { sprintf('level%d_prediction', loss_count), sprintf('level%d_label', loss_count) };
        %         loss_output = loss_name;
        %         net.addLayer(loss_name, dagnn.vllab_dag_loss('loss_type', opts.loss),loss_input, loss_output);
        net.addLayer('level1_BCE_loss', dagnn.Loss('loss', opts.loss),{'level1_prediction','level1_label'}, 'level1_BCE_loss');
        
        
        % ConvTranspose
        convtBlock=dagnn.ConvTranspose('size', [f,f,fn2,fn1], 'upsample', 2, 'crop', crop, 'hasBias', true);
        %         convtBlock=dagnn.ConvTranspose('size', [fd,fd,fn2,fn1], 'upsample', 2, 'crop', 1, 'hasBias', true);
        %         net.addLayer(['convt', num2str(ct)], convtBlock, {scale{s}}, {['x', num2str(num+1)]}, {['convt', num2str(ct), '_f'], ['convt', num2str(ct), '_b']});
        net.addLayer(['convtFuse', num2str(ctFuse_count)], convtBlock, {scale{s}}, {['x', num2str(num+1)]}, {['convtFuse', num2str(ctFuse_count), '_f'], ['convtFuse', num2str(ctFuse_count), '_b']});
        initializeLearningRateConvT(net, ['convtFuse', num2str(ctFuse_count)],sigma,f,fn2,fn1);
        
        
        % Concat
        net.addLayer(['cropCat', num2str(cropCat_count)], dagnn.Crop(), { ['x', num2str(num+1)],scale{s+1}}, {['x', num2str(num+2)]});
        net.addLayer(['concat', num2str(concat_count)], dagnn.Concat('dim', 3), {['x', num2str(num+2)], scale{s+1}}, {['x', num2str(num+3)]});
        
        % conv
        convBlock = dagnn.Conv('size', [1,1,fuse_fn,fn2], 'pad', 0, 'stride', 1, 'hasBias', true);
        net.addLayer(['conv', num2str(lyn+1)], convBlock, {['x', num2str(num+3)]}, {['x', num2str(num+4)]}, {['conv', num2str(lyn+1), '_f'], ['conv', num2str(lyn+1), '_b']});
        initializeLearningRate(net, ['conv',num2str(lyn+1)],sigma,1,fuse_fn,fn2);
        % BN
        net.addLayer(['bn', num2str(lyn+1)], dagnn.BatchNorm('numChannels', fn2), {['x', num2str(num+4)]}, {['x', num2str(num+5)]}, {['bn', num2str(lyn+1), '_f'], ['bn', num2str(lyn+1), '_b'], ['bn', num2str(lyn+1), '_m']});
        initializeLearningRateBN(net, ['bn', num2str(lyn+1)],sigma,fn2);
        % ReLU
        net.addLayer(['relu', num2str(lyn+1)], dagnn.ReLU(), {['x', num2str(num+5)]}, {['x', num2str(num+6)]});
        
        %         % conv
        %         convBlock = dagnn.Conv('size', [1,1,fuse_fn,fn2], 'pad', 0, 'stride', 1, 'hasBias', true);
        %         net.addLayer(['conv', num2str(lyn+1)], convBlock, {['x', num2str(num+5)]}, {['x', num2str(num+6)]}, {['conv', num2str(lyn+1), '_f'], ['conv', num2str(lyn+1), '_b']});
        %         initializeLearningRate(net, ['conv',num2str(lyn+1)],sigma,1,fuse_fn,fn2);
        
        num = num+6;
        lyn = lyn+1;
        %         ct = ct+1;
        ctFuse_count = ctFuse_count+1;
%         dotP_count = dotP_count+1;
        %         poolAt_count = poolAt_count+1;
        cropCat_count = cropCat_count+1;
        concat_count = concat_count+1;
        convSD_count = convSD_count+1;
        loss_count = loss_count+1;
        
        
    elseif s == 2
        %         fn1 = channel(s);
        fn1 = channel(s);
        fn2 = channel(s+1);
        fuse_fn = 2*fn2;
        
        %         % Pooling
        %         net.addLayer(['Pool_Att', num2str(poolAt_count)], dagnn.Pooling('poolSize', [4 4], 'stride', 4), {'ratio'}, {['x', num2str(num+1)]});
        
%         % Ratio Attention
%         net.addLayer(['dot_prod_' num2str(dotP_count)], dagnn.DotProduct(), {['x', num2str(num)], 'ratio_down4'}, {['x', num2str(num+1)]}) ;
        
        % 1*1 conv
        %         prediction_name = { sprintf('level%d_prediction', convSD_count) };
        convBlock = dagnn.Conv('size', [1,1,fn1,2], 'pad', 0, 'stride', 1, 'hasBias', true);
        net.addLayer(['convSD', num2str(convSD_count)], convBlock, {['x', num2str(num)]}, {['level', num2str(convSD_count),'_prediction']}, {['convSD', num2str(convSD_count), '_f'], ['convSD', num2str(convSD_count), '_b']});
        initializeLearningRate(net, ['convSD',num2str(convSD_count)],sigma,1,fn1,2);
        
        % loss layer
        %         loss_name = { sprintf('level%d_%s_loss', loss_count, opts.loss) };
        %         loss_input = { sprintf('level%d_prediction', loss_count), sprintf('level%d_label', loss_count) };
        %         loss_output = loss_name;
        %         net.addLayer(loss_name, dagnn.vllab_dag_loss('loss_type', opts.loss),loss_input, loss_output);
        net.addLayer('level2_BCE_loss', dagnn.Loss('loss', opts.loss),{'level2_prediction','level2_label'}, 'level2_BCE_loss');
        
        % ConvTranspose
        convtBlock=dagnn.ConvTranspose('size', [f,f,fn2,fn1], 'upsample', 2, 'crop', crop, 'hasBias', true);
        %         convtBlock=dagnn.ConvTranspose('size', [fd,fd,fn2,fn1], 'upsample', 2, 'crop', 1, 'hasBias', true);
        net.addLayer(['convtFuse', num2str(ctFuse_count)], convtBlock, {['x', num2str(num)]}, {['x', num2str(num+1)]}, {['convtFuse', num2str(ctFuse_count), '_f'], ['convtFuse', num2str(ctFuse_count), '_b']});
        initializeLearningRateConvT(net, ['convtFuse', num2str(ctFuse_count)],sigma,f,fn2,fn1);
        
        % Concat
        net.addLayer(['cropCat', num2str(cropCat_count)], dagnn.Crop(), { ['x', num2str(num+1)],scale{s+1}}, {['x', num2str(num+2)]});
        net.addLayer(['concat', num2str(concat_count)], dagnn.Concat('dim', 3), {['x', num2str(num+2)], scale{s+1}}, {['x', num2str(num+3)]});
        
        % conv
        convBlock = dagnn.Conv('size', [1,1,fuse_fn,fn2], 'pad', 0, 'stride', 1, 'hasBias', true);
        net.addLayer(['conv', num2str(lyn+1)], convBlock, {['x', num2str(num+3)]}, {['x', num2str(num+4)]}, {['conv', num2str(lyn+1), '_f'], ['conv', num2str(lyn+1), '_b']});
        initializeLearningRate(net, ['conv',num2str(lyn+1)],sigma,1,fuse_fn,fn2);
        % BN
        net.addLayer(['bn', num2str(lyn+1)], dagnn.BatchNorm('numChannels', fn2), {['x', num2str(num+4)]}, {['x', num2str(num+5)]}, {['bn', num2str(lyn+1), '_f'], ['bn', num2str(lyn+1), '_b'], ['bn', num2str(lyn+1), '_m']});
        initializeLearningRateBN(net, ['bn', num2str(lyn+1)],sigma,fn2);
        % ReLU
        net.addLayer(['relu', num2str(lyn+1)], dagnn.ReLU(), {['x', num2str(num+5)]}, {['x', num2str(num+6)]});
        
        
        %         ct = ct+1;
        
        lyn = lyn+1;
        num = num+6;
        ctFuse_count = ctFuse_count+1;
        cropCat_count = cropCat_count+1;
        concat_count = concat_count+1;
        %         poolAt_count = poolAt_count+1;
%         dotP_count = dotP_count+1;
        convSD_count = convSD_count+1;
        loss_count = loss_count+1;
        
    elseif s == 3
        fn1 = channel(s);
        fn2 = channel(s+1);
        fuse_fn = 2*fn2;
        
        %         % Pooling
        %         net.addLayer(['Pool_Att', num2str(poolAt_count)], dagnn.Pooling('poolSize', [2 2], 'stride', 2), {'ratio'}, {['x', num2str(num+1)]});
        
%         % Ratio Attention
%         net.addLayer(['dot_prod_' num2str(dotP_count)], dagnn.DotProduct(), {['x', num2str(num)], 'ratio_down2'}, {['x', num2str(num+1)]}) ;
        
        % 1*1 conv
        %         prediction_name = { sprintf('level%d_prediction', convSD_count) };
        convBlock = dagnn.Conv('size', [1,1,fn1,2], 'pad', 0, 'stride', 1, 'hasBias', true);
        net.addLayer(['convSD', num2str(convSD_count)], convBlock, {['x', num2str(num)]}, {['level', num2str(convSD_count),'_prediction']}, {['convSD', num2str(convSD_count), '_f'], ['convSD', num2str(convSD_count), '_b']});
        initializeLearningRate(net, ['convSD',num2str(convSD_count)],sigma,1,fn1,2);
        
        % loss layer
        %         loss_name = { sprintf('level%d_%s_loss', loss_count, opts.loss) };
        %         loss_input = { sprintf('level%d_prediction', loss_count), sprintf('level%d_label', loss_count) };
        %         loss_output = loss_name;
        %         net.addLayer(loss_name, dagnn.vllab_dag_loss('loss_type', opts.loss),loss_input, loss_output);
        net.addLayer('level3_BCE_loss', dagnn.Loss('loss', opts.loss),{'level3_prediction','level3_label'}, 'level3_BCE_loss');
        
        % ConvTranspose
        convtBlock=dagnn.ConvTranspose('size', [f,f,fn2,fn1], 'upsample', 2, 'crop', crop, 'hasBias', true);
        %         convtBlock=dagnn.ConvTranspose('size', [fd,fd,fn2,fn1], 'upsample', 2, 'crop', 1, 'hasBias', true);
        net.addLayer(['convtFuse', num2str(ctFuse_count)], convtBlock, {['x', num2str(num)]}, {['x', num2str(num+1)]}, {['convtFuse', num2str(ctFuse_count), '_f'], ['convtFuse', num2str(ctFuse_count), '_b']});
        initializeLearningRateConvT(net, ['convtFuse', num2str(ctFuse_count)],sigma,f,fn2,fn1);
        
        % Concat
        net.addLayer(['cropCat', num2str(cropCat_count)], dagnn.Crop(), { ['x', num2str(num+1)],scale{s+1}}, {['x', num2str(num+2)]});
        net.addLayer(['concat', num2str(concat_count)], dagnn.Concat('dim', 3), {['x', num2str(num+2)], scale{s+1}}, {['x', num2str(num+3)]});
        
        %         % Ratio attention
        %         %         net.addLayer(['dot_prod_' num2str(dotP_count)], dagnn.DotProduct(), {'ratio', ['x', num2str(num+3)]}, {['x', num2str(num+4)]}) ;
        %         net.addLayer(['dot_prod_' num2str(dotP_count+1)], dagnn.DotProduct(), {['x', num2str(num+4)], 'ratio'}, {['x', num2str(num+5)]}) ;
        
        % conv
%         convBlock = dagnn.Conv('size', [1,1,fuse_fn,1], 'pad', 0, 'stride', 1, 'hasBias', true);
%         net.addLayer(['conv', num2str(lyn+1)], convBlock, {['x', num2str(num+4)]}, {'prediction'}, {['conv', num2str(lyn+1), '_f'], ['conv', num2str(lyn+1), '_b']});
%         initializeLearningRate(net, ['conv',num2str(lyn+1)],sigma,1,fuse_fn,1);
        convBlock = dagnn.Conv('size', [1,1,fuse_fn,2], 'pad', 0, 'stride', 1, 'hasBias', true);
        net.addLayer(['convSD', num2str(convSD_count+1)], convBlock, {['x', num2str(num+3)]}, {['level', num2str(convSD_count+1),'_prediction']}, {['convSD', num2str(convSD_count+1), '_f'], ['convSD', num2str(convSD_count+1), '_b']});
        initializeLearningRate(net, ['convSD',num2str(convSD_count+1)],sigma,1,fuse_fn,2);        
        
        
        %         % BN
        %         net.addLayer(['bn', num2str(lyn+1)], dagnn.BatchNorm('numChannels', fuse_fn), {['x', num2str(num+3)]}, {['x', num2str(num+4)]}, {['bn', num2str(lyn+1), '_f'], ['bn', num2str(lyn+1), '_b'], ['bn', num2str(lyn+1), '_m']});
        %         initializeLearningRateBN(net, ['bn', num2str(lyn+1)],sigma,fuse_fn);
        %         % ReLU
        %         net.addLayer(['relu', num2str(lyn+1)], dagnn.ReLU(), {['x', num2str(num+4)]}, {['x', num2str(num+5)]});
        
        
        ct = ct+1;
%         lyn = lyn+1;
        num = num+3;
        cropCat_count = cropCat_count+1;
        concat_count = concat_count+1;
        %         poolAt_count = poolAt_count+1;
%         dotP_count = dotP_count+1;
        ctFuse_count = ctFuse_count+1;
        convSD_count = convSD_count+2;
        loss_count = loss_count+1;
        
    end
    
end

% % conv
% convBlock = dagnn.Conv('size', [1,1,fuse_fn,1], 'pad', 0, 'stride', 1, 'hasBias', true);
% net.addLayer(['conv', num2str(lyn+1)], convBlock, {['x', num2str(num+4)]}, {'prediction'}, {['conv', num2str(lyn+1), '_f'], ['conv', num2str(lyn+1), '_b']});
% initializeLearningRate(net, ['conv',num2str(lyn+1)],sigma,1,fuse_fn,1);

% Loss layer
% net.addLayer('loss', dagnn.vllab_dag_loss('loss_type', opts.loss),{'prediction', 'labels'}, {'loss'});
net.addLayer('level4_BCE_loss', dagnn.Loss('loss', opts.loss),{'level4_prediction','level4_label'}, 'level4_BCE_loss');

end

function initializeLearningRate(net, str_layer,sigma,f,fc,k)
ii = net.getLayerIndex(str_layer);
net.params(net.layers(ii).paramIndexes(1)).value = sigma*randn(f,f,fc,k,'single');
net.params(net.layers(ii).paramIndexes(1)).learningRate = 0.001;
net.params(net.layers(ii).paramIndexes(1)).weightDecay = 0.1;
net.params(net.layers(ii).paramIndexes(2)).value = zeros(1,k,'single');
net.params(net.layers(ii).paramIndexes(2)).learningRate = 0.001;
net.params(net.layers(ii).paramIndexes(2)).weightDecay = 0;
end

function initializeLearningRateConvT(net, str_layer,sigma,f,k,fc)
ii = net.getLayerIndex(str_layer);
net.params(net.layers(ii).paramIndexes(1)).value = sigma*randn(f,f,k,fc,'single');
net.params(net.layers(ii).paramIndexes(1)).learningRate = 0.001;
net.params(net.layers(ii).paramIndexes(1)).weightDecay = 0.1;
net.params(net.layers(ii).paramIndexes(2)).value = zeros(1,k,'single');
net.params(net.layers(ii).paramIndexes(2)).learningRate = 0.001;
net.params(net.layers(ii).paramIndexes(2)).weightDecay = 0;
end

function initializeLearningRateBN(net, str_layer,sigma,k)
ii = net.getLayerIndex(str_layer);
net.params(net.layers(ii).paramIndexes(1)).value = sigma*randn(k,1,'single');
net.params(net.layers(ii).paramIndexes(1)).learningRate = 0.001;
net.params(net.layers(ii).paramIndexes(1)).weightDecay = 0.1;
net.params(net.layers(ii).paramIndexes(2)).value = zeros(k,1,'single');
net.params(net.layers(ii).paramIndexes(2)).learningRate = 0.001;
net.params(net.layers(ii).paramIndexes(2)).weightDecay = 0;
net.params(net.layers(ii).paramIndexes(3)).value = zeros(k,2,'single');
net.params(net.layers(ii).paramIndexes(3)).learningRate = 0.001;
net.params(net.layers(ii).paramIndexes(3)).weightDecay = 0;
end