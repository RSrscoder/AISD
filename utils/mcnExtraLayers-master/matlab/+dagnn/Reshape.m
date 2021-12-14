classdef Reshape < dagnn.ElementWise
  properties
    shape = {} ;
  end

  properties (Transient)
    inputSizes = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnreshape(inputs{1}, obj.shape) ;
      obj.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnreshape(inputs{1}, obj.shape, derOutputs{1}) ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      tmp = zeros(inputSizes{1}) ;
      reshaped = vl_nnreshape(tmp, obj.shape) ;
      outSz = ones(1,4) ;
      outSz(1:numel(size(reshaped))) = size(reshaped) ;
      outputSizes{1} = outSz ;
    end

    function rfs = getReceptiveFields(obj)
    end

    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      load@dagnn.Layer(obj, s) ;
    end

    function obj = Reshape(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
