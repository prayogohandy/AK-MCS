function cumprob=queryCDF(dist,x)
% wrapper function for querying CDF
nmcs=size(x,1);
cumprob=zeros(nmcs,size(dist,1));
for n=1:nmcs
    if mod(n,round(nmcs/100)*10)==0 && nmcs>1e5
        fprintf('%f\n',n/nmcs*100);
    end
    for i=1:size(dist,1)
        param1=dist{i,2};
        param2=dist{i,3};
        switch dist{i,1}
            case 'norm'
                p=normcdf(x(n,i),param1,param2);
            case 'logn'
                p=logncdf(x(n,i),param1,param2);
            case 'gumbel'
                p=evcdf(x(n,i),param1,param2);
            case 'gumbel-'
                p=evcdf(-x(n,i),-param1,param2);
            case 'unif'
                p=unifcdf(x(n,i),param1,param2);
            case 'wbl'
                p=wblcdf(x(n,i),param1,param2);
        end
        cumprob(n,i)=p;
    end
end