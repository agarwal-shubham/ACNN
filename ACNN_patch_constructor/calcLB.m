function [M, DiagS] = calcLB(shape, params)
% The L-B operator matrix is computed by DiagS^-1*M.

H = shearMat(shape, params.an, params.nbinsl);

% Calculate the weights matrix M
M = calcCotMatrixM1([shape.X, shape.Y, shape.Z], shape.TRIV, H, params.nbinsl);

% Calculate the diagonal of matrix S 
DiagS = calcVoronoiRegsCircCent(shape.TRIV, [shape.X, shape.Y, shape.Z]);
%%
DiagS = abs( DiagS );
%%

end

% ----------------------------------------------------------------------- %
function [H] = shearMat(shape, anisotropy, theta_bins)
	
	H = zeros(size(shape.TRIV, 1), theta_bins, 3, 3);
	aniso = [ anisotropy,0,0 ; 0,1,0 ; 0,0,1 ];
	[U1, U2, D, normalf] = compute_diffusion_tensor([shape.X, shape.Y, shape.Z], shape.TRIV);
	
	for k = 1:size(shape.TRIV, 1)
		U = [U1(:,k), U2(:,k), normalf(:,k)]; %[Umax,Umin,normal] all are normalized already
		for kk = 1:theta_bins
			theta = 360*(kk-1)/theta_bins;		%solve for isotropic case (theta = 0) separately later
			% theta = 360*kk/(theta_bins+1);
			Rot = RotMatrix(theta);
			H(k,kk,:,:) = Rot*U*aniso*U'*Rot';
		end
	end
end

% ----------------------------------------------------------------------- %
function [Rot] = RotMatrix(theta)

	Rotx = [ 1,0,0 ; 0,cosd(theta),-1*sind(theta) ; 0,sind(theta),cosd(theta) ];
	Roty = [ cosd(theta),0,sind(theta) ; 0,1,0 ; -1*sind(theta),0,cosd(theta) ];
	Rotz = [ cosd(theta),-1*sind(theta),0 ; sind(theta),cosd(theta),0 ; 0,0,1 ];
	Rot = Rotx*Roty*Rotz;
end

% ----------------------------------------------------------------------- %
function [M] = calcCotMatrixM1(Vertices, Tri, H, theta_bins)

N = size(Vertices, 1);
M = cell(1,theta_bins);

v1 = Vertices(Tri(:, 2), :) - Vertices(Tri(:, 1), :);
v2 = Vertices(Tri(:, 3), :) - Vertices(Tri(:, 1), :);
v3 = Vertices(Tri(:, 3), :) - Vertices(Tri(:, 2), :);

% sin = sqrt(1-cos^2);  cos = (a.b)/(|a|*|b|)
tmp1 = dot( v1,  v2, 2); sin1 = sqrt( 1 - ( ((tmp1).^2)./(normVec(v1).^2.*normVec(v2).^2) ) ); clear tmp1;
tmp2 = dot(-v1,  v3, 2); sin2 = sqrt( 1 - ( ((tmp2).^2)./(normVec(v1).^2.*normVec(v3).^2) ) ); clear tmp2;
tmp3 = dot(-v2, -v3, 2); sin3 = sqrt( 1 - ( ((tmp3).^2)./(normVec(v2).^2.*normVec(v3).^2) ) ); clear tmp3;

%normalize edges (cant do before finding angles)
v1 = v1./repmat(normVec(v1), 1, 3);
v2 = v2./repmat(normVec(v2), 1, 3);
v3 = v3./repmat(normVec(v3), 1, 3);

for kk=1:theta_bins
	MM = sparse(N,N);
	for k = 1:size(Tri, 1) %over all faces
	    MM(Tri(k, 1), Tri(k, 2)) = MM(Tri(k, 1), Tri(k, 2)) + ( squeeze(v3(k,:)) * squeeze(H(k,kk,:,:)) * squeeze(v2(k,:))' )/sin3(k);
	    MM(Tri(k, 1), Tri(k, 3)) = MM(Tri(k, 1), Tri(k, 3)) + ( (-squeeze(v3(k,:))) * squeeze(H(k,kk,:,:)) * squeeze(v1(k,:))' )/sin2(k);
	    MM(Tri(k, 2), Tri(k, 3)) = MM(Tri(k, 2), Tri(k, 3)) + ( (-squeeze(v2(k,:))) * squeeze(H(k,kk,:,:)) * (-squeeze(v1(k,:)))' )/sin1(k);
	end
	MM = 0.5*(MM + MM'); % here she does the normalization (comment - Artiom)
	    
	% inds = sub2ind([N, N], [Tri(:, 2); Tri(:, 1); Tri(:, 1)], [Tri(:, 3); Tri(:, 3); Tri(:, 2)]);
	% M(inds) = M(inds) + [cot1; cot2; cot3];
	% inds = sub2ind([N, N], [Tri(:, 3); Tri(:, 3); Tri(:, 2)], [Tri(:, 2); Tri(:, 1); Tri(:, 1)]);
	% M(inds) = M(inds) + [cot1; cot2; cot3];
	% M = 0.5*(M + M');
	% % M(M < 0) = 0;

	MM = MM - diag(sum(MM, 2)); % making it Laplacian
	M{kk} = MM;
	clear MM;
end

    function normV = normVec(vec)
        normV = sqrt(sum(vec.^2, 2));
    end
%     function normalV = normalizeVec(vec)
%         normalV = vec./repmat(normVec(vec), 1, 3);
%     end

end

% % ----------------------------------------------------------------------- %
% function [M] = calcCotMatrixM(Vertices, Tri, H) %#ok<DEFNU>

% N = size(Vertices, 1);
% [transmat] = calcTransmat(N, Tri);

% % Calculate the matrix M, when {M}_ij = (cot(alpha_ij) + cot(beta_ij))/2
% % Calculate the matrix M, when {M}_ij = [(e(kj)'*H*e(ki))/sin(alpha_ij)] + [(e(hj)'*H*e(hi))/sin(beta_ij)]
% % [transrow, transcol] = find(triu(transmat,1) > 0);
% [transrow, transcol] = find((triu(transmat,1) > 0) | (triu(transmat',1) > 0));
% M = sparse(N, N);

% for k = 1:length(transrow)
    
%     P = transrow(k);
%     Q = transcol(k);
%     S = transmat(P,Q);
%     R = transmat(Q,P);
%     % P,Q is the common edge i,j
%     % R is k and S is h
%     %alpha = angle ikj or PRQ
%     %beta = angle ihj or PSQ 
%     %%
% %     u1 = Vertices(Q, :) - Vertices(R, :); u1 = u1./norm(u1);
% %     v1 = Vertices(P, :) - Vertices(R, :); v1 = v1./norm(v1);
% %     u2 = Vertices(P, :) - Vertices(S, :); u2 = u2./norm(u2);
% %     v2 = Vertices(Q, :) - Vertices(S, :); v2 = v2./norm(v2);
% %     M(P,Q) = -1/2*(dot(u1, v1)/norm(cross(u1, v1)) + dot(u2, v2)/norm(cross(u2, v2)));

%     tmp1 = 0;
%     tmp2 = 0;
    
%     if (R ~= 0)		%alpha
%         u1 = Vertices(Q, :) - Vertices(R, :); u1 = u1./norm(u1);
%         v1 = Vertices(P, :) - Vertices(R, :); v1 = v1./norm(v1);
%         sin_alpha = sqrt(1-(dot(u1, v1)^2));
%     end

%     if (S ~= 0)		%beta
%         u2 = Vertices(P, :) - Vertices(S, :); u2 = u2./norm(u2);
%         v2 = Vertices(Q, :) - Vertices(S, :); v2 = v2./norm(v2);
%         sin_beta = sqrt(1-(dot(u2, v2)^2));
%     end
    
%     M(P,Q) = -1/2*(tmp1 + tmp2);
%     %%
    
% end

% M = 0.5*(M + M');
% M = M - diag(sum(M, 2));

% end


% % ----------------------------------------------------------------------- %
% function [transmat] = calcTransmat(N, Tri)

% % Calculation of the map of all the connected vertices: for each i,j,
% % transmat(i,j) equals to the third vertex of the triangle which connectes
% % them; if the vertices aren't connected - transmat(i,j) = 0.
% transmat = sparse(N, N);
% transmat(sub2ind(size(transmat), Tri(:,1), Tri(:,2))) = Tri(:,3);
% transmat(sub2ind(size(transmat), Tri(:,2), Tri(:,3))) = Tri(:,1);
% transmat(sub2ind(size(transmat), Tri(:,3), Tri(:,1))) = Tri(:,2);

% end