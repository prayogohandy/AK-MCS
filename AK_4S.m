clc;close all;
%% Add lfunc before running
lfunc='U';

%% Algorithm Parameter
N1=15; % initial DoE size
nmcs=1e4; 
switch lfunc
    case 'U'
        learnThres=-2;
    case 'EFF'
        learnThres=1e-3;
    case 'H'
        learnThres=0.1;
    case 'ERF'
        learnThres=1e-4;
end

%% Problem Parameter
gfunc=@fourSeriesProb;
tdist={'norm',0,1 % w KN/m
    'norm',0,1 % E KN/m2
    };
dist=tdist;
d=size(dist,1);
theta = ones(1,size(dist,1)); 
lob = ones(1,size(dist,1))*1e-2; upb = ones(1,size(dist,1))*1e2;
regfunc=@regpoly0; corrfunc=@corrgauss;

%% Generate Sampling Space
fprintf('Generating Sampling Space\n')
tic
xt=monteCarlo(dist,nmcs);
t1=toc;

%% Generate DoE
fprintf('Generating DoE\n')
tic
xd=monteCarlo(dist,N1);
gx=gfunc(xd);
t2=toc;

%% Train Kriging
fprintf('Train Kriging\n')
tic
dmodel = dacefit(xd, gx, regfunc, corrfunc, theta, lob, upb);
t3=toc;

iter=0;
touter=0;tlearn=0;tupdate=0;outer=0;idxt=true(nmcs,1);
%% Outer Loop
while true
    outer=outer+1;
    %% Inner Loop
    nxt=size(xt,1);
    oldidxt=idxt;
    idxt=true(nxt,1); % initiate idx
    idxt(1:length(oldidxt))=oldidxt; % copy previous idxt
    while true
        iter=iter+1;
        fprintf('Iteration: %d\n',iter);
        %% Finding Best Point using Learning Function
        fprintf('Finding Best Point\n');
        tic;
        realindex=1:nxt;realindex=realindex(idxt); % tracking idx     
        [ymat,varmat]=predictor(xt, dmodel); 
        switch lfunc 
            case 'U'
                [xstar,learn,idxs]=learnU(xt(idxt,:),ymat(idxt,:),varmat(idxt,:));
                learn=-learn; % to make threshold minimize
            case 'EFF'
                [xstar,learn,idxs]=learnEFF(xt(idxt,:),ymat(idxt,:),varmat(idxt,:));
            case 'H'
                [xstar,learn,idxs]=learnH(xt(idxt,:),ymat(idxt,:),varmat(idxt,:));
            case 'ERF'
                [xstar,learn,idxs]=learnERF(xt(idxt,:),ymat(idxt,:),varmat(idxt,:));
        end
        gstar=gfunc(xstar);
        
        %% Stopping Condition
        fprintf('Learning Threshold: %.5f\n of %.5f\n',learn,learnThres);
        if learn<=learnThres
            break
        end
        tlearn(iter)=toc;
        
        %% Update DoE and Kriging Model
        fprintf('Update DoE and Kriging Model\n');
        tic;
        idxt(realindex(idxs))=false; % deactivate sample already in DoE
        xd=[xd;xstar];
        gx=[gx;gstar];
        dmodel = dacefit(xd, gx, regfunc, corrfunc, theta, lob, upb);
        tupdate(iter)=toc;
    end
    
    %% Calculate Pf
    % disable current layer for next active learning
    tic;
    ysign=ymat<=0;
    Pf1=mean(ysign);
    fprintf('Pf: %.3e\n',Pf1);
    cov=sqrt((1-Pf1)/Pf1/nmcs);
    if cov<0.05
        break
    end
    nreq=(1-Pf1)/Pf1/0.05^2;
    if Pf1~=0
        nadd=ceil((nreq-nmcs)/1e4)*1e4;
    else
        nadd=1e4;
    end
    nmcs=size(xt,1);
    xt=[xt;monteCarlo(dist,nadd)];
    touter(outer)=toc;
    if outer > 20
        break
    end
end
t4=toc;
fprintf('Total Evaluated Sample   %d\n',size(xd,1));
fprintf('Generate Sampling Space: %.4f\n',t1)
fprintf('Generate DoE:            %.4f\n',t2)
fprintf('Train Kriging:           %.4f\n',t3)
fprintf('Learn Kriging:           %.4f\n',sum(tlearn))
fprintf('Update Kriging:          %.4f\n',sum(tupdate))
fprintf('Outer Loop:              %.4f\n',sum(touter))
fprintf('Last Pf:                 %.4f\n',t4)
ttotal=t1+t2+t3+t4+sum(touter)+sum(tlearn)+sum(tupdate);
fprintf('Total Time:              %.4f\n',ttotal)

function [xstar,minU,idxs,U] = learnU(S,ymat,varmat)
U=zeros(size(ymat,1),1);
for i=1:size(ymat,1)
    if mod(i,ceil(size(ymat,1)/100)*10)==0
        fprintf('%f\n',i/size(ymat,1)*100);
    end
    y=ymat(i);var=varmat(i);
    U(i,:)=abs(y)/sqrt(var);
end
[minU,idxs]=min(U);
xstar=S(idxs,:);
end

function [xstar,maxEFF,idxs,EFF] = learnEFF(S,ymat,varmat)
EFF=zeros(size(ymat,1),1);
for i=1:size(ymat,1)
    if mod(i,ceil(size(ymat,1)/100)*10)==0
        fprintf('%f\n',i/size(ymat,1)*100);
    end
    y=ymat(i);var=varmat(i);
    sig=sqrt(var);
    Gmin=-2*sig; Gplus=2*sig;
    EFF(i,:)=y*(2*normcdf(0,y,sig)-normcdf(Gmin,y,sig)-normcdf(Gplus,y,sig))...
        -sig*(2*normpdf(0,y,sig)-normpdf(Gmin,y,sig)-normpdf(Gplus,y,sig))...
        +2*sig*(normcdf(Gplus,y,sig)-normcdf(Gmin,y,sig));
end
[maxEFF,idxs]=max(EFF);
xstar=S(idxs,:);
end

function [xstar,maxH,idxs,H] = learnH(S,ymat,varmat)
H=zeros(size(ymat,1),1);
for i=1:size(ymat,1)
    if mod(i,ceil(size(ymat,1)/100)*10)==0
        fprintf('%f\n',i/size(ymat,1)*100);
    end
    y=ymat(i);var=varmat(i);
    sig=sqrt(var);
    Gmin=-2*sig; Gplus=2*sig;
    a=(2*sig-y)/2; b=(2*sig+y)/2;
    H(i,:)=...
    abs(...
        log(sqrt(2*pi)*sig+0.5)*(normcdf(Gplus,y,sig)-normcdf(Gmin,y,sig))...
        -(a*normpdf(Gplus,y,sig)+b*normpdf(Gmin,y,sig))...
    );
end
[maxH,idxs]=max(H);
xstar=S(idxs,:);
end

function [xstar,maxERF,idxs,ERF] = learnERF(S,ymat,varmat)
ERF=zeros(size(ymat,1),1);
for i=1:size(ymat,1)
    if mod(i,ceil(size(ymat,1)/100)*10)==0
        fprintf('%f\n',i/size(ymat,1)*100);
    end
    y=ymat(i);sig=sqrt(varmat(i));
    ERF(i,:)=-sign(y)*y*normcdf(-sign(y)*y/sig)+sig*normpdf(y/sig);
end
[maxERF,idxs]=max(ERF);
xstar=S(idxs,:);
end
