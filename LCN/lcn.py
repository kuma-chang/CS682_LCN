from builtins import range
import numpy as np

from cs682.layers import *

def lcn_forward(x, w, b, lcn_param):
    """
    Computes the forward pass for an locally-connected layer.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - w: Weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - lcn_param: A dictionary with the following keys:
      - 'center_dist': The distance between input center and weight center.

    Returns a tuple of:
    - out: Output data, of shape (N, F, HH, WW)
    - cache: (x, w, b, input_index, weight_index, lcn_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the locally-connected forward pass.                     #
    ###########################################################################
    center_dist = lcn_param['center_dist']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    out_H = HH
    out_W = WW
    out = np.zeros((N, F, out_H, out_W))

    input_index_h = np.linspace(H//2, -(H//2), H)
    input_index_w = np.linspace(-(W//2), W//2, W)
    input_index = np.zeros((H,W,2))
    input_index[:, :].T[0] = input_index_h
    input_index[:, :, 1] = input_index_w

    weight_index_h = np.linspace(HH//2, -(HH//2), HH)
    weight_index_w = np.linspace(-(WW//2), WW//2, WW)
    weight_index = np.zeros((HH,WW,2))
    weight_index[:, :].T[0] = weight_index_h
    weight_index[:, :, 1] = weight_index_w

    for n in range(N):
        for f in range(F):
            for i in range(out_H):
                for j in range(out_W):
                    #Calculate the distance mask
                    input_dist_mask = input_index - weight_index[i][j]
                    input_dist_mask = np.sum(np.square(input_dist_mask), axis=2) + center_dist**2

                    #Calculate the score with out the distance penalty
                    out_temp = (x[n].reshape(C,H*W).T*w[f,:,i,j]).T
                    out_temp = out_temp.reshape(C,H,W)
                    out_temp = np.sum(out_temp, axis=0)

                    #Adding the distance penalty
                    out[n,f,i,j] = np.sum(out_temp/input_dist_mask) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, input_index, weight_index, lcn_param)
    return out, cache


def lcn_backward(dout, cache):
    """
    A naive implementation of the backward pass for a locally-connected layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, input_index, weight_index, lcn_param)
        - x: Input data of shape (N, C, H, W)
        - w: Weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - lcn_param: A dictionary with the following keys:
          - 'center_dist': The distance between input center and weight center.

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, input_index, weight_index, lcn_param = cache
    
    center_dist = lcn_param['center_dist']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    out_H = dout.shape[2] 
    out_W = dout.shape[3] 
    
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    for n in range(N):
        for f in range(F):
            for i in range(out_H):
                for j in range(out_W):
                    #Calculate the distance mask
                    input_dist_mask = input_index - weight_index[i][j]
                    input_dist_mask = np.sum(np.square(input_dist_mask), axis=2) + center_dist**2

                    dx_temp = np.zeros(dx[n].shape)
                    dx_temp += dout[n, f, i, j]/input_dist_mask
                    dx_temp = ((dx_temp).reshape(C,H*W).T * w[f,:,i,j]).T.reshape(C,H,W)
                    dx[n] += dx_temp

                    dw_temp = np.zeros(dx[n].shape)
                    dw_temp += dout[n, f, i, j]/input_dist_mask
                    dw_temp = dw_temp * x[n]
                    dw_temp = dw_temp.reshape(C,H*W)
                    dw_temp = np.sum(dw_temp, axis=1)
                    dw[f,:,i,j] += dw_temp

                    db[f] += dout[n, f, i, j]
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def lcn_relu_forward(x, w, b, lcn_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, lcn_cache = lcn_forward(x, w, b, lcn_param)
    out, relu_cache = relu_forward(a)
    cache = (lcn_cache, relu_cache)
    return out, cache


def lcn_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    lcn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = lcn_backward(da, lcn_cache)
    return dx, dw, db

