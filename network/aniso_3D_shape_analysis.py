
# coding: utf-8

# # Prerequisites
# Install Theano and Lasagne using the following commands:
# 
# ```bash
# pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
# pip install https://github.com/Lasagne/Lasagne/archive/master.zip
# ```

# ## Prepared data
# 
# All it is required to train on the FAUST_registration dataset (for GCNN patches) for this demo is available for download at
# https://www.dropbox.com/s/aamd98nynkvbcop/EG16_tutorial.tar.bz2?dl=0

# # ICNN Toolbox
# 
# ```bash
# git clone https://github.com/jonathanmasci/EG16_tutorial.git
# ```
# 
# ![](http://www.people.usi.ch/mascij/EG16_tutorial/shapenet_architecture.png)

# In[1]:

import sys
import os
import numpy as np
import scipy.io
import time

import theano
import theano.tensor as T
import theano.sparse as Tsp

import lasagne as L
import lasagne.layers as LL
import lasagne.objectives as LO
from lasagne.layers.normalization import batch_norm

sys.path.append('..')
from icnn import aniso_utils_lasagne, dataset, snapshotter


# ## Data loading

# In[2]:

base_path = '../ACNN_patch_constructor/data/train/patch/'

# train_txt, test_txt, descs_path, patches_path, geods_path, labels_path, ...
        # desc_field='desc', patch_field='M', geod_field='geods', label_field='labels', epoch_size=100
ds = dataset.ClassificationDatasetPatchesMinimal(
    'FAUST_registrations_train.txt', 'FAUST_registrations_test.txt',
    os.path.join(base_path, 'descs', 'shot'),
    os.path.join(base_path, 'patch_aniso', 'alpha=100_nangles=016_ntvals=005_tmin=6.000_tmax=24.000_thresh=99.900_norm=L1'), 
    None, 
    os.path.join(base_path, 'labels'),
    epoch_size=50)


# In[3]:

# inp = LL.InputLayer(shape=(None, 544))
# print(inp.input_var)
# patch_op = LL.InputLayer(input_var=Tsp.csc_fmatrix('patch_op'), shape=(None, None))
# print(patch_op.shape)
# print(patch_op.input_var)
# icnn = LL.DenseLayer(inp, 16)
# print(icnn.output_shape)
# print(icnn.output_shape)
# desc_net = theano.dot(patch_op, icnn)


# ## Network definition

# In[3]:

nin = 544
nclasses = 6890
l2_weight = 1e-5

def get_model(inp, patch_op):
    icnn = LL.DenseLayer(inp, 16)
    icnn = batch_norm(aniso_utils_lasagne.ACNNLayer([icnn, patch_op], 16, nscale=5, nangl=16))
    icnn = batch_norm(aniso_utils_lasagne.ACNNLayer([icnn, patch_op], 32, nscale=5, nangl=16))
    icnn = batch_norm(aniso_utils_lasagne.ACNNLayer([icnn, patch_op], 64, nscale=5, nangl=16))
    ffn = batch_norm(LL.DenseLayer(icnn, 512))
    ffn = LL.DenseLayer(icnn, nclasses, nonlinearity=aniso_utils_lasagne.log_softmax)

    return ffn

inp = LL.InputLayer(shape=(None, nin))
patch_op = LL.InputLayer(input_var=Tsp.csc_fmatrix('patch_op'), shape=(None, None))

ffn = get_model(inp, patch_op)

# L.layers.get_output -> theano variable representing network
output = LL.get_output(ffn)
pred = LL.get_output(ffn, deterministic=True)  # in case we use dropout

# target theano variable indicatind the index a vertex should be mapped to wrt the latent space
target = T.ivector('idxs')

# to work with logit predictions, better behaved numerically
cla = aniso_utils_lasagne.categorical_crossentropy_logdomain(output, target, nclasses).mean()
acc = LO.categorical_accuracy(pred, target).mean()

# a bit of regularization is commonly used
regL2 = L.regularization.regularize_network_params(ffn, L.regularization.l2)


cost = cla + l2_weight * regL2


# ## Define the update rule, how to train

# In[4]:

params = LL.get_all_params(ffn, trainable=True)
grads = T.grad(cost, params)
# computes the L2 norm of the gradient to better inspect training
grads_norm = T.nlinalg.norm(T.concatenate([g.flatten() for g in grads]), 2)

# Adam turned out to be a very good choice for correspondence
updates = L.updates.adam(grads, params, learning_rate=0.001)


# ## Compile

# In[5]:

funcs = dict()
funcs['train'] = theano.function([inp.input_var, patch_op.input_var, target],
                                 [cost, cla, l2_weight * regL2, grads_norm, acc], updates=updates,
                                 on_unused_input='warn')
print("dbg1")
funcs['acc_loss'] = theano.function([inp.input_var, patch_op.input_var, target],
                                    [acc, cost], on_unused_input='warn')
funcs['predict'] = theano.function([inp.input_var, patch_op.input_var],
                                   [pred], on_unused_input='warn')


# # Training (a bit simplified)

# In[6]:

n_epochs = 1
eval_freq = 1

start_time = time.time()
best_trn = 1e5
best_tst = 1e5

kvs = snapshotter.Snapshotter('demo_training.snap')

for it_count in xrange(n_epochs):
    tic = time.time()
    b_l, b_c, b_s, b_r, b_g, b_a = [], [], [], [], [], []
    for x_ in ds.train_iter():
        print("dbg2")
        tmp = funcs['train'](*x_)
        print("dbg3")
        # do some book keeping (store stuff for training curves etc)
        b_l.append(tmp[0])
        b_c.append(tmp[1])
        b_r.append(tmp[2])
        b_g.append(tmp[3])
        b_a.append(tmp[4])
    epoch_cost = np.asarray([np.mean(b_l), np.mean(b_c), np.mean(b_r), np.mean(b_g), np.mean(b_a)])
    print(('[Epoch %03i][trn] cost %9.6f (cla %6.4f, reg %6.4f), |grad| = %.06f, acc = %7.5f %% (%.2fsec)') %
                 (it_count, epoch_cost[0], epoch_cost[1], epoch_cost[2], epoch_cost[3], epoch_cost[4] * 100, 
                  time.time() - tic))

    if np.isnan(epoch_cost[0]):
        print("NaN in the loss function...let's stop here")
        break
    print("dbg4")
        
    if (it_count % eval_freq) == 0:
        v_c, v_a = [], []
        for x_ in ds.test_iter():
            tmp = funcs['acc_loss'](*x_)
            v_a.append(tmp[0])
            v_c.append(tmp[1])
        test_cost = [np.mean(v_c), np.mean(v_a)]
        print(('           [tst] cost %9.6f, acc = %7.5f %%') % (test_cost[0], test_cost[1] * 100))

        if epoch_cost[0] < best_trn:
            kvs.store('best_train_params', [it_count, LL.get_all_param_values(ffn)])
            best_trn = epoch_cost[0]
        if test_cost[0] < best_tst:
            kvs.store('best_test_params', [it_count, LL.get_all_param_values(ffn)])
            best_tst = test_cost[0]
print("...done training %f" % (time.time() - start_time))


# # Test phase
# Now that the model is train it is enough to take the fwd function and apply it to new data.

# In[7]:

rewrite = True

out_path = '/tmp/EG16_tutorial/dumps/' 
print "Saving output to: %s" % out_path

if not os.path.isdir(out_path) or rewrite==True:
    try:
        os.makedirs(out_path)
    except:
        pass
    
    a = []
    for i,d in enumerate(ds.test_iter()):
        fname = os.path.join(out_path, "%s" % ds.test_fnames[i])
        print fname,
        tmp = funcs['predict'](d[0], d[1])[0]
        a.append(np.mean(np.argmax(tmp, axis=1).flatten() == d[2].flatten()))
        scipy.io.savemat(fname, {'desc': tmp})
        print ", Acc: %7.5f %%" % (a[-1] * 100.0)
    print "\nAverage accuracy across all shapes: %7.5f %%" % (np.mean(a) * 100.0)
else:
    print "Model predictions already produced."


# # Results

# ![](http://www.people.usi.ch/mascij/EG16_tutorial/shapenet_corr.png)
