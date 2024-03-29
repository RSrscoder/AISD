function net = AddBilinearUpSampling(net, var_to_up_sample, upsampled_var, upsample_fac, opts)
filters = single(bilinear_u(upsample_fac*2, opts.num_classes, opts.num_classes)) ;
% filters = single(bilinear_u(upsample_fac*2, 1, 1)) ;
% filters = single(bilinear_kernel(upsample_fac*2, opts.num_classes, opts.num_classes)) ;
% filters = single(bilinear_kernel(4, 1, 1)) ;
crop = upsample_fac/2;
deconv_name = ['dec_' upsampled_var];
net.addLayer(deconv_name, ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', upsample_fac, ...
    'crop', [crop crop crop crop], ...
    'opts', {'cudnn','nocudnn'}, ...
    'numGroups', opts.num_classes, ...
    'hasBias', false), ...
    var_to_up_sample, upsampled_var, [deconv_name 'f']) ;
f = net.getParamIndex([deconv_name 'f']) ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;
% net.params(f).trainMethod = 'nothing';

