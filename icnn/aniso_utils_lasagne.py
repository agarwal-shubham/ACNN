"""
Some basic helpers operating on Lasagne
"""
from collections import OrderedDict
import numpy as np

import theano
import theano.tensor as T

import lasagne as L
import lasagne.layers as LL
import lasagne.nonlinearities as LN
import lasagne.init as LI
import lasagne.updates as LU
import logging



def L2_dist_squared(x, y):
    xsq = T.sqr(x).sum(axis=1).reshape((x.shape[0], 1))
    ysq = T.sqr(y).sum(axis=1).reshape((1,y.shape[0]))
    return xsq + ysq - 2.0 * T.dot(x, y.T) + 1E-06


class ACNNLayer(LL.MergeLayer):
    """
    """
    def __init__(self, incomings, nfilters, nscale=5, nangl=16,
                 W=LI.GlorotNormal(), b=LI.Constant(0.0),
                 normalize_scale=False, normalize_input=False,
                 nonlinearity=L.nonlinearities.rectify, **kwargs):
        super(ACNNLayer, self).__init__(incomings, **kwargs)
        self.nfilters = nfilters
        self.filter_shape = (nfilters, self.input_shapes[0][1], nscale, nangl)
        # output_shape, input_shape, scales, angles eg.-> 64,32,5,16
        print("self.filter_shape", self.filter_shape)
        self.nscale = nscale
        self.nangl = nangl
        self.normalize_scale = normalize_scale
        self.normalize_input = normalize_input
        self.nonlinearity = nonlinearity
        # input is multiplied by W
        self.W = self.add_param(W, self.filter_shape, name="W") # shape of W is filter_shape
        biases_shape = (nfilters, )
        self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        shp = input_shapes[0]
        nangl = self.nangl
        nscale = self.nscale
        # out_shp = (shp[0], self.nfilters * nscale(flattened) * nangl)
        out_shp = (shp[0], self.nfilters * 1 * nangl)			# debug4
        # print("shp[0]", shp[0])
        print("out_shp", out_shp)
        return out_shp

    def get_output_for(self, inputs, **kwargs):
        y, M = inputs
        #[y, M] is [icnn, patch] most probably

        if self.normalize_input:    #L2norm; default is false
            y /= T.sqrt(T.sum(T.sqr(y), axis=1) + 1e-5).dimshuffle(0, 'x')

        # theano.dot works both for sparse and dense matrices
        desc_net = theano.dot(M, y)

        # y is icnn so its shape[1] is the #filters in prev layer
        desc_net = T.reshape(desc_net, (M.shape[1], self.nscale, self.nangl, y.shape[1]))
        desc_net = desc_net.dimshuffle(0, 3, 1, 2)
        print("desc_net_reshaped_shuffled",self.input_shapes[0][0], self.filter_shape[1], self.nscale, self.nangl)
        #batch_size, channels, rows, cols
        
        if self.normalize_scale:    #default is false
            # Unit length per scale
            desc_net /= (1e-5 + T.sqrt(T.sum(T.sqr(desc_net), axis=2) + 1e-5).dimshuffle(0, 1, 'x', 2))

        # pad it along the angl axis so that conv2d produces circular
        # convolution along that dimension
        desc_net = T.concatenate([desc_net, desc_net[:, :, :, :-1]], axis=3)        # debug5

        # output is N x outmaps x 1 x nangl if filter size is the same as input image size prior padding
        # input to conv2d is (input, filters, input_shape, filter_shape)
        y = theano.tensor.nnet.conv.conv2d(desc_net, self.W, 
            (self.input_shapes[0][0], self.filter_shape[1], self.nscale, self.nangl * 2 - 1), self.filter_shape)
        # output is of shape (batch size, output channels, output rows, output columns)
        

        if self.b is not None:
            y += self.b.dimshuffle('x', 0, 'x', 'x')

        y = y.flatten(2)	# reduces y to 2 dimns keeping the 1st dimn intact and collapsing all others

        return self.nonlinearity(y)


class COVLayer(LL.Layer):
    def __init__(self, incoming, **kwargs):
        super(COVLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])

    def get_output_for(self, input, **kwargs):
        x = input
        x -= x.mean(axis=0)
        x = T.dot(x.T, x) / (self.input_shape[0] - 1)
        x = x.flatten(2)
        return x


def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))


def categorical_crossentropy_logdomain(log_predictions, targets, nclasses):
    # return -T.sum(theano.tensor.extra_ops.to_one_hot(targets, nclasses) * log_predictions, axis=1)
    # http://deeplearning.net/tutorial/logreg.html#logreg
    return - log_predictions[T.arange(targets.shape[0]), targets]
