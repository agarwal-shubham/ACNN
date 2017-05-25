function [M] = aniso_patch(shape, f, lbo, params)

	% parameters
	thr		=	params.thr; 
	nbinssc	=	params.nbinssc; 
	nbinsl	=	params.nbinsl;
	ranget	=	params.rangesc;
	tmin	=	ranget(1);
	tmax	=	ranget(2);
	stept	=	(tmax - tmin)/(nbinssc-1);

	% [desc,shape]	=	signature(shape,'wks');

	% M	=	cell(1,size(shape.X,1));	%6890 cells
	M	=	cell(1,603-597+1);	%6890 cells
	ids = 0:nbinsl*nbinssc:(size(shape.X,1)-1)*nbinsl*nbinssc;
	
	[M{:}] = deal(sparse(zeros(1,nbinsl*nbinssc*size(shape.X,1))));		%initialize each cell element to zero matrix
	for t = tmin:stept:tmax
		idx = ((t-tmin)/stept);
		% heatKer = arrayfun(@(ind) lbo.Phi{ind}*diag(exp(-t*lbo.Lambda{ind}))*lbo.Phi{ind}', 1:size(lbo.Lambda,2), 'UniformOutput', false);
		% heatKer = arrayfun(@(ind) lbo.Phi{ind}*diag(exp(-t*lbo.Lambda{ind}/norm(lbo.Lambda{ind})))*lbo.Phi{ind}', 1:size(lbo.Lambda,2), 'UniformOutput', false);
		% ^ too much in program memory required
		% denr = arrayfun(@(ind) sum(heatKer{ind}(i,:)), 1:size(heatKer,2), 'UniformOutput', false);
		for l = 1:nbinsl
			iter = (idx*nbinsl) + l;
			iter
			phi		=	lbo.Phi{l};
			lambda	=	lbo.Lambda{l};
			%should eigenval be normalized?
			lambda	=	lambda/norm(lambda);
			% eigenvec form orthonormal basis, so should be normalised
			% phi		=	phi./repmat(sqrt(sum(phi.^2,2)),1,size(phi,2));
			heatK	=	phi*diag(exp(-t*lambda))*phi';	%t=6-24 leads to all 0 diag values of exp(...) if lambda is not normalized
			% for i = 1:size(shape.X,1)
			for i = 597:603
				M_i = zeros(1,nbinsl*nbinssc*size(shape.X,1));
				denmr	=	sum(heatK(i,:));
				% heatK	=	heatKer{l};
				% denmr	=	denr{l};

				% tmp		=	( repmat(heatK(i,:)',1,size(f,2)) .* f ) / denmr ;		% (rep((1 X 6890)X544)) .* (6890 X 544) = (6890 X 544)
				% take row wise L1 norm , o/p is a 6890*1 col matrix
				% M_i(ids + (idx*nbinsl) + l) = arrayfun(@(ind) norm(tmp(ind,:),1), 1:size(tmp,1));
				
				tmp		=	( repmat(heatK(i,:)',1,size(f,2)) .* f );		% (rep((1 X 6890)X544)) .* (6890 X 544) = (6890 X 544)
				tmp(tmp<thr) = 0;
				M_i(ids + iter) = arrayfun(@(ind) norm(tmp(ind,:),1)/denmr, 1:size(tmp,1));
				M{i-596}	=	M{i-596} + sparse(M_i);
				clear M_i;
				
				% clear tmp;
			end
		end
	end