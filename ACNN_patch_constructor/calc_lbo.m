function [Evecs,Evals,area] = calc_lbo(shape, params)

[W, A] = calcLB(shape, params);
area = diag(A);
Evecs = cell(1,params.nbinsl);
Evals = cell(1,params.nbinsl);
%should A or W be normalized
% A = A/norm(full(A));
for k=1:params.nbinsl
	
	% w = W{k}/norm(W{k},1);
	% [evecs,evals] = eigs(w, a, params.nLBO, -1e-5, struct('disp', 0));	%eigen value soln for W.Phi = A.Phi.Lambda

	[evecs,evals] = eigs(squeeze(W{k}), A, params.nLBO, -1e-5, struct('disp', 0));	%eigen value soln for W.Phi = A.Phi.Lambda
	evals = abs(diag(real(evals)));

	[evals, perm] = sort(evals);
	evecs = real(evecs(:, perm));
	Evecs{k} = evecs;
	Evals{k} = evals;
end