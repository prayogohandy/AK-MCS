function prob=queryPDF(dist,x)
% wrapper function for querying PDF
nmcs=size(x,1);
prob=zeros(nmcs,size(dist,1));
for n=1:nmcs
    if mod(n,round(nmcs/100)*10)==0 && nmcs>1e5
        fprintf('%f\n',n/nmcs*100);
    end
    for i=1:size(dist,1)
        param1=dist{i,2};
        param2=dist{i,3};
        switch dist{i,1}
            case 'norm'
                p=normpdf(x(n,i),param1,param2);
            case 'logn'
                p=lognpdf(x(n,i),param1,param2);
            case 'gumbel'
                p=evpdf(x(n,i),param1,param2);
            case 'gumbel-'
                p=evpdf(-x(n,i),-param1,param2);
            case 'unif'
                p=unifpdf(x(n,i),param1,param2);
            case 'wbl'
                p=wblpdf(x(n,i),param1,param2);
        end
        prob(n,i)=p;
    end
end