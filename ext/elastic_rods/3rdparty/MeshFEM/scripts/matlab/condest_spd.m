function [c, v] = condest_spd(A,t)
%CONDEST 1-norm condition number estimate.
%   C = CONDEST(A) computes a lower bound C for the 1-norm condition
%   number of a square matrix A.
%
%   C = CONDEST(A,T) changes T, a positive integer parameter equal to
%   the number of columns in an underlying iteration matrix.  Increasing the
%   number of columns usually gives a better condition estimate but increases
%   the cost.  The default is T = 2, which almost always gives an estimate
%   correct to within a factor 2.
%
%   [C,V] = CONDEST(A) also computes a vector V which is an approximate null
%   vector if C is large.  V satisfies NORM(A*V,1) = NORM(A,1)*NORM(V,1)/C.
%
%   Note: CONDEST uses random numbers generated by RAND.  If repeatable
%   results are required,  use RNG to control MATLAB's random number
%   generator state.
%
%   CONDEST is based on the 1-norm condition estimator of Hager [1] and a
%   block oriented generalization of Hager's estimator given by Higham and
%   Tisseur [2].  The heart of the algorithm involves an iterative search
%   to estimate ||A^{-1}||_1 without computing A^{-1}. This is posed as the
%   convex, but nondifferentiable, optimization problem: 
%
%         max ||A^{-1}x||_1 subject to ||x||_1 = 1. 
%
%   See also NORMEST1, COND, NORM, RAND.

%   Reference:
%   [1] William W. Hager, Condition estimates, 
%       SIAM J. Sci. Stat. Comput. 5, 1984, 311-316, 1984.
% 
%   [2] Nicholas J. Higham and Fran\c{c}oise Tisseur, 
%       A Block Algorithm for Matrix 1-Norm Estimation 
%       with an Application to 1-Norm Pseudospectra, 
%       SIAM J. Matrix Anal. App. 21, 1185-1201, 2000. 
%
%   Nicholas J. Higham
%   Copyright 1984-2011 The MathWorks, Inc.
% Julian Panetta: this function based on MATLAB's condest function uses Cholesky
% factorization instead of LU, and thus works much more efficiently on SPD
% matrices.

if size(A,1) ~= size(A,2)
   error(message('MATLAB:condest:NonSquareMatrix'))
end
if isempty(A), c = 0; v = []; return, end
if nargin < 2, t = []; end

[L,~, ~] = chol(A, 'lower', 'vector');

% Unfortunately, MATLAB doesn't intelligently compute L' \ b:
% it creates a transposed copy of L for each solve
% For this, we would need to install cs_utsolve from SuiteSparse.
% To make things more efficient, we do the transpose once (though it wastes
% memory).
U = L';
k = find(abs(diag(L))==0);
warns = warning('query','all');
temp = onCleanup(@()warning(warns));
warning('off','all');
if ~isempty(k)
   c = Inf;
   n = length(A);
   v = zeros(n,1);
   k = min(k);
   v(k) = 1;
   if k > 1
    % v(1:k-1) = -U(1:k-1,1:k-1)\U(1:k-1,k);
      v(1:k-1) = -L(1:k-1,1:k-1)\U(k,1:k-1);
   end
else
   [Ainv_norm, ~, v] = normest1(@condestf,t);
   A_norm = norm(A,1);
   c = Ainv_norm*A_norm;
end
v = v/norm(v,1);

    function f = condestf(flag, X)
        %CONDESTF   Function used by CONDEST.        
        if isequal(flag,'dim')
            f = max(size(L));
        elseif isequal(flag,'real')
            f = isreal(L);
        elseif isequal(flag,'notransp')
            f = U\(L\X);
        elseif isequal(flag,'transp')
            f = U\(L\X);
        end
    end

end


