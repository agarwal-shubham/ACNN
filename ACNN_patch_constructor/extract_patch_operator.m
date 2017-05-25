function extract_patch_operator(srcpath, featpath, lbopath, dstpath, patch_params)

    if ~exist(dstpath, 'dir')
        mkdir(dstpath);
    end

    fnames = dir(fullfile(srcpath, '*.mat'));
    parfor i = 1 : length(fnames)
        if exist(fullfile(dstpath, fnames(i).name), 'file')
            fprintf('%s already processed, skipping\n', fnames(i).name)
            continue
        end
        fprintf('Processing %s\n', fnames(i).name)
        tmp = load(fullfile(srcpath, fnames(i).name));
        shape = tmp.shape;
        tmp = load(fullfile(lbopath, fnames(i).name));
        lbo = tmp;
	%use SHOT feature descriptor as used in the paper
	%other alternatives include wks and isc        
	tmp = load(fullfile(featpath, fnames(i).name));
        f = tmp.desc;
        % clear tmp;    %messes up with parallel pool

        M = aniso_patch(shape, f, lbo, patch_params);
        % make a big matrix out of all the various M_i cells
        % each matrix in the cell array is stacked row after row.
        % this allows a more efficient multiplication and handling in theano
        M = sparse(cat(1, M{:}))';
        
        parsave(fullfile(dstpath, fnames(i).name), M);
    end
end

function parsave(fn, M)
    save(fn, 'M', '-v7.3')
end
