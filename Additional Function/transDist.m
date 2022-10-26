function xt=transDist(x,targetDist)
% transform distribution from standard normal to targetDist
cdf=normcdf(x);
xt=queryICDF(targetDist,cdf);
end