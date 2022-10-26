function x=monteCarlo(dist,nmcs)
cum=rand(nmcs,size(dist,1));
x=queryICDF(dist,cum);
end