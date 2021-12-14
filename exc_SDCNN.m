function [output, time] = exc_SDCNN(input, net, gpu)
% -------------------------------------------------------------------------
%   Description:
%       function to apply SDCNN
%
%   Input:
%       - input : original shadow image
%       - net   : SDCNN model
%       - gpu   : GPU ID
%
%   Output:
%       - output: shadow detection result
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

    %% setup
    net.mode = 'test' ;
    output_var = 'prediction';
    output_index = net.getVarIndex(output_var);
    net.vars(output_index).precious = 1;
       
    % convert to GPU
    y = single(input);
    
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
           
end