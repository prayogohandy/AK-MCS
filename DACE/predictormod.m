function  [y, or1, or2, dmse] = predictormod(x, dmodel)
or1 = NaN;   or2 = NaN;  dmse = NaN;  % Default return values
q = 1;  % number of response functions
if  isnan(dmodel.beta)
error('DMODEL has not been found')
end

[m,n] = size(dmodel.S);  % number of design sites and number of dimensions
sx = size(x);            % number of trial sites and their dimension
if  min(sx) == 1 && n > 1 % Single trial point 
nx = max(sx);
if  nx == n 
  mx = 1;  x = x(:).';
end
else
mx = sx(1);  nx = sx(2);
end
if  nx ~= n
error(sprintf('Dimension of trial sites should be %d',n));
end

% Get distances to design sites  
dx = zeros(mx*m,n);  kk = 1:m;
for  k = 1 : mx
dx(kk,:) = repmat(x(k,:),m,1) - dmodel.S;
kk = kk + m;
end
% Get regression function and correlation
f = feval(dmodel.regr, x);
r = feval(dmodel.corr, dmodel.theta, dx);
r = reshape(r, m, mx);

% Scaled predictor 
y = f * dmodel.beta + (dmodel.gamma * r).';

if  nargout > 1   % MSE wanted
rt = dmodel.C \ r;
u = dmodel.G \ (dmodel.Ft.' * rt - f.');
or1 = repmat(dmodel.sigma2,mx,1) .* repmat((1 + colsum(u.^2) - colsum(rt.^2))',1,q);
if  nargout > 2
disp('WARNING from PREDICTOR.  Only  y  and  or1=mse  are computed')
end
end
  
% >>>>>>>>>>>>>>>>   Auxiliary function  ====================

function  s = colsum(x)
% Columnwise sum of elements in  x
if  size(x,1) == 1,  s = x; 
else,                s = sum(x);  end