rng(42,'twister')
addpath(genpath('isc'))
getd = @(p)path(p,path);
getd('toolbox_signal/');
getd('toolbox_general/');
getd('toolbox_graph/');

%% Compute LBO
lbo_params.nLBO = 300;			% #eigen-values = k
lbo_params.nbinsl = 16;			% # theta bins for different laplacians
lbo_params.an = 100;			% anisotropy
lbo_params.nbinssc = 5;			% #bins for scale
lbo_params.rangesc = [6,24];			% range for scale
extract_lbo('data/train/shapes/', 'data/train/lbo', lbo_params);
% extract_lbo('data/test/shapes/', 'data/test/lbo', lbo_params);

%% Compute patch operator
patch_params.nbinssc	= 5;		% number of scales
patch_params.rangesc	= [6,24];	% range for scale
patch_params.nbinsl		= 16;		% number of angles
patch_params.thr		= 99.900;	% for correspondence matching
extract_patch_operator('data/train/shapes', 'data/train/descs', 'data/train/lbo', 'data/train/patch', patch_params);
% extract_patch_operator('data/test/shapes', 'data/test/descs', 'data/test/lbo', 'data/test/patch', patch_params);
