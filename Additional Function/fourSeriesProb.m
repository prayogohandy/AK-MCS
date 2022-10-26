function gx = fourSeriesProb(x)
%     https://www.sciencedirect.com/science/article/pii/S0167473020300035
    x1=x(:,1);
    x2=x(:,2);
    g(:,1)=3+(x1-x2).^2./10-(x1+x2)./sqrt(2);
    g(:,2)=3+(x1-x2).^2./10+(x1+x2)./sqrt(2);
    g(:,3)=x1-x2+7/sqrt(2);
    g(:,4)=x2-x1+7/sqrt(2);
    gx=min(g,[],2);
end