function extract_lbo(srcpath, dstpath, lbo_params)

	fnames = dir(fullfile(srcpath, '*.mat'));
	minSc = lbo_params.rangesc(1);
	maxSc = lbo_params.rangesc(2);
	binsSc = lbo_params.nbinssc;
	stp  = (maxSc-minSc)/(binsSc-1);
	parfor i = 1 : length(fnames)
		fprintf('Processing %s\n', fnames(i).name)
		tmp = load(fullfile(srcpath, fnames(i).name));
		[Phi, Lambda, A] = calc_lbo(tmp.shape, lbo_params);
		% parsave(fullfile(dstpath, strcat('phi_lambda_A_16_', fnames(i).name)), Phi, Lambda, A);
		parsave(fullfile(dstpath, fnames(i).name), Phi, Lambda, A);

		% hk = cell(1,lbo_params.nbinsl);
		% for k=1:lbo_params.nbinsl
		% 	phi = squeeze(Phi{k});
		% 	lambda = squeeze(Lambda{k});
		% 	theta = 360*(k-1)/lbo_params.nbinsl;
		% 	hk_tvals = cell(1,binsSc);
		% 	for kk = 1:binsSc
		% 		t = minSc + ( (kk-1)*stp )
		% 		tmp = phi*exp(-t.*diag(lambda))*phi';		%heat kernel
		% 		hk_tvals{kk} = sparse(tmp);
		% 	end
		% 	hk{k} = hk_tvals;
		% 	% if ~exist(fullfile(dstpath, strcat('lapl',num2str(k)), strcat('tval', num2str(kk))), 'dir')
		% 	% 	mkdir(fullfile(dstpath, strcat('lapl',num2str(k)), strcat('tval', num2str(kk))));
		% 	% end
		% end
		% parsave(fullfile(dstpath, strcat('kernel_lapl16_t5_', fnames(i).name)), hk);
	end
end

function parsave(fn, Phi, Lambda, A)
	switch nargin
		case 4
			save(fn, 'Phi', 'Lambda', 'A', '-v7.3')
		case 2
			hk = Phi;
			save(fn, 'hk', '-v7.3');
	end
end