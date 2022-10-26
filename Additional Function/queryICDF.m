function x=queryICDF(dist,cum)
% wrapper function for querying ICDF
nmcs=size(cum,1);
x=zeros(nmcs,size(dist,1));
for n=1:nmcs
    if mod(n,round(nmcs/100)*10)==0 && nmcs>1e5
        fprintf('%f\n',n/nmcs*100);
    end
    for i=1:size(dist,1)
        param1=dist{i,2};
        param2=dist{i,3};
        switch dist{i,1}
            case 'norm'
                p=norminv(cum(n,i),param1,param2);
            case 'logn'
                p=logninv(cum(n,i),param1,param2);
            case 'gumbel'
                p=evinv(cum(n,i),param1,param2); % minima
            case 'gumbel-'
                p=-evinv(1-cum(n,i),-param1,param2); % maxima
            case 'unif'
                p=unifinv(cum(n,i),param1,param2);
            case 'wbl'
                p=wblinv(cum(n,i),param1,param2);
        end
        x(n,i)=p;
    end
end