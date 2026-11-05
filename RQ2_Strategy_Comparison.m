%% Initialization
clc;
clearvars;

%% Configuration
cfg=struct();
cfg.dataFiles={'BCC.csv','CEV.csv','ERA.csv','LEV.csv','MMG.csv', ...
    'HS.csv','NJN.csv','NZ.csv','SI.csv'};
cfg.scriptDir=fileparts(mfilename('fullpath'));
if isempty(cfg.scriptDir), cfg.scriptDir=pwd; end
cfg.dataDir=fullfile(cfg.scriptDir,'Datasets');
cfg.outputDir=cfg.scriptDir;
cfg.texName=[mfilename,'.tex'];
cfg.poolCount=50;
cfg.batchSize=3;
cfg.rounds=3;
cfg.refPerClass=3;
cfg.numRepeats=30;
cfg.L=3;
cfg.Ns=2000;
cfg.numTrajectories=10;
cfg.etaSigma=0.5;
cfg.xi=0.4;
cfg.epsilonAssignment=1e-6;
cfg.epsilonPhi=1e-5;
cfg.adequacyNormalization='recommendation_max';
cfg.splitSeed=20261009;
cfg.chainSeed=271828;
cfg.feedbackSeed=314159;
cfg.randomSeed=141421;
cfg.bootstrapSeed=161803;
cfg.bootstrapReplicates=10000;
cfg.confidenceLevel=0.95;
cfg.workers=16;
cfg.useParallel=true;
cfg.nChains=4;
cfg.burnIn=100;
cfg.chainDraws=200000;
cfg.usePreconditioning=true;
cfg.preconditioningMethod='hessian';
cfg.directionStrategy='coordinate_mixture';
cfg.sphereProbability=0.2;
cfg.solverTolerance=1e-9;
cfg.feasibilityTolerance=1e-9;
cfg.lpRetryMaxIterations=100000;
cfg.lpFallbackAlgorithm='interior-point';
cfg.lpFallbackMaxIterations=10000;
cfg.minimumRadius=1e-12;
cfg.lineIterations=60;
cfg.barrierMaxIterations=80;
cfg.barrierNewtonTolerance=1e-9;
cfg.barrierArmijo=0.01;
cfg.barrierStepFraction=0.99;
cfg.cacheStates=16;
cfg.modelCacheMB=256;
cfg.useMexSampler=true;
cfg.verbose=false;
cfg.displayDetails=false;
cfg.progressEvery=1000;
cfg.queue=[];
cfg.coreTieTol=1e-12;
cfg.liAlpha=0.1;
cfg.baselineTieSeed=173205;
cfg.methods={'AVG-TP-TC','Random','HCAI','HAIW','MS-Raw','SM','ES','LS','Core-set-VP'};

%% Validation and Execution
assert(exist('linprog','file')==2,'Optimization Toolbox with linprog is required.');
assert(isfinite(cfg.coreTieTol) && cfg.coreTieTol>=0);
assert(cfg.liAlpha>0 && cfg.liAlpha<1);
addpath(cfg.scriptDir);
assert(exist('har_chord_mex','file')==3,'Comparison:MissingMex', ...
    'Compiled har_chord_mex is required. Place the MEX binary for this MATLAB platform beside this script.');
assert(cfg.numRepeats>=2 && cfg.poolCount>=cfg.rounds*cfg.batchSize);
assert(cfg.xi>0 && cfg.xi<0.5 && cfg.etaSigma>0 && cfg.epsilonPhi>0);
assert(mod(cfg.Ns,cfg.nChains)==0 && mod(cfg.chainDraws,cfg.Ns/cfg.nChains)==0);
oldRng=rng; rngGuard=onCleanup(@()rng(oldRng));
protocolTests(cfg);
inputs=cell(numel(cfg.dataFiles),1);
for d=1:numel(inputs)
    path=fullfile(cfg.dataDir,cfg.dataFiles{d});
    assert(exist(path,'file')==2,'Missing CSV: %s',path);
    [X,y,preprocessing]=readData(path);
    [~,name]=fileparts(path);
    [ref,poolIDs,testIDs]=splitData(y,cfg.refPerClass,cfg.poolCount);
    fprintf('%s: N=%d, classes=%d, reference=%d, initial pool=%d, fixed test=%d.\n', ...
        name,numel(y),max(y),numel(ref),numel(poolIDs),numel(testIDs));
    inputs{d}=struct('X',X,'y',y,'Dataset',name,'Preprocessing',preprocessing);
end
if cfg.useParallel
    assert(license('test','Distrib_Computing_Toolbox'),'Parallel Toolbox required.');
    pool=gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers~=cfg.workers, delete(pool); pool=[]; end
    if isempty(pool)
        cluster=parcluster('local'); cluster.NumWorkers=max(cluster.NumWorkers,cfg.workers);
        pool=parpool(cluster,cfg.workers);
    end
    addAttachedFiles(pool,{which('har_chord_mex')});
    queue=parallel.pool.DataQueue;
else
    queue=[];
end
nD=numel(inputs); nJ=nD*cfg.numRepeats;
totalRounds=nJ*numel(cfg.methods)*cfg.rounds;
progressReport([],totalRounds);
if cfg.useParallel, afterEach(queue,@(m)progressReport(m,totalRounds)); end
wall=tic; jobs=cell(nJ,1);
if cfg.useParallel
    parfor j=1:nJ
        d=floor((j-1)/cfg.numRepeats)+1; r=mod(j-1,cfg.numRepeats)+1;
        jobs{j}=runMainRepeat(inputs{d},d,r,cfg,queue,totalRounds);
    end
else
    for j=1:nJ
        d=floor((j-1)/cfg.numRepeats)+1; r=mod(j-1,cfg.numRepeats)+1;
        jobs{j}=runMainRepeat(inputs{d},d,r,cfg,queue,totalRounds);
    end
end
summary=summarizeMain(jobs,nD,cfg);
summary.WallSeconds=toc(wall);
writeMainTex(fullfile(cfg.outputDir,cfg.texName),summary,inputs,cfg);

%% Paired Experiment
function job=runMainRepeat(input,d,r,cfg,queue,totalRounds)
splitSeed=keySeed(cfg.splitSeed,d,r,'split'); rng(splitSeed,'twister');
[ref,pool,test]=splitData(input.y,cfg.refPerClass,cfg.poolCount);
ref=sort(ref); pool=sort(pool); test=sort(test);
X=input.X; y=input.y; P=basis(X,cfg.L); q=max(y);
context=struct('DatasetID',d,'Repeat',r,'X',X,'P',P,'q',q, ...
    'Universe',pool,'nCriteria',size(X,2));
Dcov=sqrt(max(0,sum((reshape(X(pool,:),numel(pool),1,[])- ...
    reshape(X(pool,:),1,numel(pool),[])).^2,3)/size(X,2)));
positive=Dcov(triu(true(numel(pool)),1)&Dcov>0);
if isempty(positive), sigmaCov=eps; else, sigmaCov=median(positive); end
context.Kcov=exp(-(Dcov/sigmaCov).^2);
context.SigmaCov=sigmaCov;
nM=numel(cfg.methods); T=cfg.rounds;
job=struct('Dataset',input.Dataset,'Repeat',r,'SplitSeed',splitSeed, ...
    'ReferenceIDs',ref,'PoolIDs',pool,'TestIDs',test,'SigmaCov',sigmaCov);
job.OMAE=zeros(nM,T); job.InitialOMAE=zeros(nM,1);
job.RoundSeconds=zeros(nM,T); job.SelectionSeconds=zeros(nM,T);
job.TrajectorySeconds=zeros(nM,T); job.EvaluationSeconds=zeros(nM,T);
job.InitialSeconds=zeros(nM,1); job.TotalSeconds=zeros(nM,1);
job.QueryIDs=cell(nM,T); job.StateSeeds=cell(nM,T+1);
job.TrajectoryUnionSize=nan(nM,T); job.FallbackUsed=false(nM,T);
job.RegionCounts=nan(nM,T,3); job.NumericalFailures=zeros(nM,1);
methodOrder=circshift(1:nM,[0,mod((d-1)*cfg.numRepeats+r-1,nM)]);
for m=methodOrder
    methodClock=tic; localCfg=cfg; localCfg.progressQueue=queue;
    localCfg.progressMethod=cfg.methods{m};
    ctx=context;
    ctx.ModelCache=containers.Map('KeyType','char','ValueType','any');
    ctx.ModelCache('__meta__')=struct('Order',{{}},'Bytes',0,'Hits',0);
    cache=containers.Map('KeyType','char','ValueType','any'); order={};
    reference=ref; labels=y(ref); active=pool;
    initialClock=tic;
    [base,order]=getState(reference,labels,ctx,localCfg,cache,order,true,'none');
    model=mean(base.Models,2);
    job.InitialOMAE(m)=predictionError(model,P(test,:),y(test),base.Region);
    job.InitialSeconds(m)=toc(initialClock); job.StateSeeds{m,1}=base.Seeds;
    for t=1:T
        roundClock=tic; selectionClock=tic;
        if m==1
            base=ensureEvidence(base,'TP',ctx,localCfg);
            cache(base.StateKey)=base;
            state=mainState(base,active,ctx,localCfg);
            job.RegionCounts(m,t,:)=reshape([sum(state.Action==1), ...
                sum(state.Action==2),sum(state.Action==3)],1,1,3);
            [query,order,diagnosis]=mainBatch(state,reference,labels,active, ...
                ctx,localCfg,cache,order,t);
            job.TrajectorySeconds(m,t)=diagnosis.TrajectorySeconds;
            job.TrajectoryUnionSize(m,t)=diagnosis.UnionSize;
            job.FallbackUsed(m,t)=diagnosis.Fallback;
        else
            query=competitorBatch(cfg.methods{m},base,reference,active,ctx,localCfg,t);
        end
        job.SelectionSeconds(m,t)=toc(selectionClock);
        assert(numel(query)==min(cfg.batchSize,numel(active)) && ...
            numel(unique(query))==numel(query) && all(ismember(query,active)));
        assert(~any(ismember(query,test)),'Test sample entered query batch.');
        job.QueryIDs{m,t}=query;
        reference=[reference;query]; labels=[labels;y(query)];
        [reference,ix]=sort(reference); labels=labels(ix);
        active=setdiff(active,query,'stable');
        evaluationClock=tic;
        [base,order]=getState(reference,labels,ctx,localCfg,cache,order,true,'none');
        model=mean(base.Models,2);
        job.OMAE(m,t)=predictionError(model,P(test,:),y(test),base.Region);
        job.StateSeeds{m,t+1}=base.Seeds;
        job.EvaluationSeconds(m,t)=toc(evaluationClock);
        job.RoundSeconds(m,t)=toc(roundClock);
        message=struct('Kind','round','Count',1,'Dataset',input.Dataset, ...
            'Repeat',r,'Method',cfg.methods{m},'Round',t);
        if isempty(queue), progressReport(message,totalRounds); else, send(queue,message); end
    end
    job.TotalSeconds(m)=toc(methodClock);
end
job.AOMAE=mean(job.OMAE,2);
end

%% Prediction Error
function e=predictionError(model,P,y,R)
pred=assignClasses(P*model(1:R.p),model(R.p+1:R.nVar),R.q);
e=mean(abs(pred-y))/(R.q-1);
end

%% Three-Way Partition
function state=mainState(base,active,context,cfg)
[found,rows]=ismember(active,base.IDs); assert(all(found));
model=mean(base.Models,2); R=base.Region;
U=context.P(active,:)*model(1:R.p);
hat=assignClasses(U,model(R.p+1:R.nVar),R.q);
score=adequacyScore(base.TP(rows,:),hat,cfg);
b=[0;model(R.p+1:R.nVar);1]; lo=b(hat); hi=b(hat+1);
width=hi-lo; mac=ones(size(U)); valid=width>0;
mac(valid)=1-2*min(U(valid)-lo(valid),hi(valid)-U(valid))./width(valid);
mac=max(0,min(1,mac));
risks=[mac.*(1-score),cfg.xi*((1-mac).*score+mac.*(1-score)),(1-mac).*score];
[~,action]=min(risks,[],2);
alpha=((1-cfg.xi)*mac)./((1-cfg.xi)*mac+cfg.xi*(1-mac));
state=struct('IDs',active,'Rows',rows,'Score',score,'Alpha',alpha, ...
    'Action',action,'RAD',base.RAD(rows),'Base',base);
end

%% Trajectory Coverage
function [query,order,info]=mainBatch(state,ref,labels,active,context,cfg,cache,order,roundID)
k=min(cfg.batchSize,numel(active)); negative=find(state.Action==3);
info=struct('Fallback',numel(negative)<k,'UnionSize',NaN,'TrajectorySeconds',0);
if numel(negative)<k
    ranking=sortrows([-state.RAD(negative),state.IDs(negative)],[1,2]);
    query=ranking(:,2);
    boundary=find(state.Action==2); take=min(k-numel(query),numel(boundary));
    if take>0
        if take==numel(boundary), added=sort(state.IDs(boundary));
        else
            [~,ix]=ismember(state.IDs(boundary),context.Universe);
            weights=normalizedWeights(max(0,state.Alpha(boundary)-state.Score(boundary)));
            added=greedyCoverage(context.Kcov(ix,ix),weights,take,state.IDs(boundary));
        end
        query=[query;added];
    end
    remaining=k-numel(query);
    if remaining>0
        positive=find(state.Action==1);
        ranking=sortrows([state.Score(positive),-state.RAD(positive),state.IDs(positive)],[1,2,3]);
        query=[query;ranking(1:remaining,3)];
    end
    return;
end
trajectoryClock=tic;
eligible=sort(state.IDs(negative));
[~,eligiblePoolRows]=ismember(eligible,context.Universe);
demandIDs=zeros(cfg.numTrajectories,k); importance=zeros(cfg.numTrajectories,k);
kernel=zeros(numel(eligible),cfg.numTrajectories*k);
stream=RandStream('mt19937ar','Seed',keySeed(cfg.feedbackSeed, ...
    context.DatasetID,context.Repeat,sprintf('round=%d',roundID)));
uniformInputs=rand(stream,cfg.numTrajectories,k);
for trajectory=1:cfg.numTrajectories
    tmpRef=ref; tmpLabels=labels; visited=[];
    for position=1:k
        [base,order]=getState(tmpRef,tmpLabels,context,cfg,cache,order,false,'TP');
        remaining=setdiff(eligible,visited,'stable');
        [~,rows]=ismember(remaining,base.IDs);
        ranked=sortrows([-base.RAD(rows),remaining],[1,2]); chosen=ranked(1,2);
        [~,row]=ismember(chosen,base.IDs);
        demandIDs(trajectory,position)=chosen;
        importance(trajectory,position)=base.RAD(row);
        [~,demandPoolRow]=ismember(chosen,context.Universe);
        column=(trajectory-1)*k+position;
        kernel(:,column)=context.Kcov(eligiblePoolRows,demandPoolRow);
        cumulative=cumsum(base.TP(row,:)); cumulative(end)=1;
        hypothetical=find(uniformInputs(trajectory,position)<cumulative,1);
        tmpRef=[tmpRef;chosen]; tmpLabels=[tmpLabels;hypothetical]; visited=[visited;chosen];
    end
end
universe=unique(demandIDs(:)); [~,universeRows]=ismember(universe,eligible);
weights=zeros(cfg.numTrajectories*k,1);
for trajectory=1:cfg.numTrajectories
    weights((trajectory-1)*k+(1:k))=normalizedWeights(importance(trajectory,:)');
end
info.TrajectorySeconds=toc(trajectoryClock); info.UnionSize=numel(universe);
query=greedyCoverage(kernel(universeRows,:),weights,k,universe);
end

%% Baseline Selection
function query=competitorBatch(method,base,ref,active,context,cfg,roundID)
k=min(cfg.batchSize,numel(active)); [~,rows]=ismember(active,base.IDs);
support=base.MS(rows,:);
switch method
    case 'Random'
        stream=RandStream('mt19937ar','Seed',keySeed(cfg.randomSeed, ...
            context.DatasetID,context.Repeat,sprintf('round=%d',roundID)));
        ix=randperm(stream,numel(active),k); query=active(ix); query=query(:); return;
    case 'HCAI'
        term=zeros(size(support)); positive=support>0;
        term(positive)=support(positive).*log2(support(positive));
        score=-sum(term,2);
    case 'HAIW'
        z=classMaximumMargins(base.Region,context.P(active,:),cfg);
        possible=z>=-cfg.feasibilityTolerance;
        assert(all(any(possible,2)),'Comparison:NoPossibleClass', ...
            'HAIW found no possible class; inspect solver tolerances.');
        classes=1:context.q;
        low=min(possible.*classes+(~possible)*infValue(context.q),[],2);
        high=max(possible.*classes,[],2);
        score=high-low+1;
    case {'MS-Raw','SM','ES','LS'}
        z=liHypotheticalObjectives(base.Region,context.P(active,:),cfg);
        score=liAcquisitionScore(z,method);
    case 'Core-set-VP'
        F=base.U/sqrt(size(base.Models,2));
        [found,refRows]=ismember(ref,base.IDs); assert(all(found));
        query=codeCoreSet(F(refRows,:),F(rows,:),k,active,cfg);
        return;
    otherwise, error('Unknown competitor %s.',method);
end
stream=RandStream('mt19937ar','Seed',keySeed(cfg.baselineTieSeed, ...
    context.DatasetID,context.Repeat,sprintf('ties-round=%d',roundID)));
keys=rand(stream,numel(context.Universe),1);
[found,positions]=ismember(active,context.Universe); assert(all(found));
ranking=sortrows([-score,keys(positions),(1:numel(active))'],[1,2,3]);
query=active(ranking(1:k,3));
end

%% Class Bound
function value=infValue(q)
value=q+1;
end

%% Acquisition Scores
function score=liAcquisitionScore(z,method)
prob=exp(z-max(z,[],2)); prob=prob./sum(prob,2);
switch method
    case 'SM', score=sum(z,2);
    case 'MS-Raw'
        ordered=sort(z,2,'descend');
        score=ordered(:,2)-ordered(:,1);
    case 'ES'
        term=zeros(size(prob)); positive=prob>0;
        term(positive)=prob(positive).*log(prob(positive));
        score=-sum(term,2);
    case 'LS', score=1-max(prob,[],2);
    otherwise, error('Comparison:UnknownScore','Unknown Li score %s',method);
end
assert(all(isfinite(score)),'Nonfinite acquisition score.');
end

%% Hypothetical Assignment Optimization
function Z=liHypotheticalObjectives(R,P,cfg)
p=R.p; nv=R.nVar; n=numel(R.y)+1; ne=nv+2*n+1;
margin=ne; f=zeros(ne,1); f(nv+(1:2*n))=(1-cfg.liAlpha)/n;
f(margin)=-cfg.liAlpha;
structA=[R.S,zeros(size(R.S,1),2*n+1)]; structB=R.sb;
for h=2:R.q-1
    row=zeros(1,ne); row(p+h-1)=1; row(p+h)=-1; row(margin)=1;
    structA(end+1,:)=row; structB(end+1,1)=0;
end
Aeq=[R.eq,zeros(1,2*n+1)];
lb=[-inf(nv,1);zeros(2*n,1);cfg.epsilonAssignment];
ub=[inf(nv+2*n,1);1/(R.q-1)];
numRows=sum(R.y>1)+sum(R.y<R.q);
refA=zeros(numRows,ne); cursor=0;
for j=1:n-1
    if R.y(j)>1
        cursor=cursor+1; refA(cursor,1:p)=-R.Phi(j,:);
        refA(cursor,p+R.y(j)-1)=1; refA(cursor,nv+j)=-1;
    end
    if R.y(j)<R.q
        cursor=cursor+1; refA(cursor,1:p)=R.Phi(j,:);
        refA(cursor,p+R.y(j))=-1; refA(cursor,nv+n+j)=-1;
        refA(cursor,margin)=1;
    end
end
fixedA=sparse([structA;refA]); fixedRhs=[structB;zeros(numRows,1)];
Aeq=sparse(Aeq);
templates=cell(R.q,1); utilityRows=cell(R.q,1);
for h=1:R.q
    extra=zeros((h>1)+(h<R.q),ne); cursor=0; signs=[];
    if h>1
        cursor=cursor+1; extra(cursor,p+h-1)=1;
        extra(cursor,nv+n)=-1; signs(end+1,1)=-1;
    end
    if h<R.q
        cursor=cursor+1; extra(cursor,p+h)=-1;
        extra(cursor,nv+2*n)=-1; extra(cursor,margin)=1;
        signs(end+1,1)=1;
    end
    templates{h}=extra; utilityRows{h}=signs;
end
Z=zeros(size(P,1),R.q);
for i=1:size(P,1)
    for h=1:R.q
        extra=templates{h}; extra(:,1:p)=utilityRows{h}*P(i,:);
        A=[fixedA;sparse(extra)]; rhs=[fixedRhs;zeros(size(extra,1),1)];
        try
            [~,objective]=solveLP(f,A,rhs,Aeq,1,lb,ub,cfg);
        catch ME
            error('Comparison:LiHypotheticalLP', ...
                'Hypothetical LP row %d class %d failed: %s',i,h,ME.message);
        end
        Z(i,h)=-objective;
    end
end
end

%% Class Feasibility Optimization
function Z=classMaximumMargins(R,P,cfg)
n=numel(R.y); nBase=R.nVar+n; N=size(P,1); Z=zeros(N,R.q);
A0=[R.lpA,zeros(size(R.lpA,1),1)];
eq=[R.lpEq,0]; lb=[R.lpLb;-inf];
f=zeros(nBase+1,1); f(end)=-1;
for i=1:N
    for h=1:R.q
        extra=zeros(0,nBase+1); rhs=zeros(0,1);
        if h>1
            row=zeros(1,nBase+1); row(1:R.p)=-P(i,:);
            row(R.p+h-1)=1; row(end)=1;
            extra(end+1,:)=row; rhs(end+1,1)=-R.epsilon;
        end
        if h<R.q
            row=zeros(1,nBase+1); row(1:R.p)=P(i,:);
            row(R.p+h)=-1; row(end)=1;
            extra(end+1,:)=row; rhs(end+1,1)=0;
        end
        try
            [solution,~,~]=solveLP(f,[A0;extra],[R.lpRhs;rhs],eq,1,lb,[],cfg);
        catch ME
            error('Comparison:MarginLP', ...
                'Class-margin LP failed at candidate row %d, class %d: %s',i,h,ME.message);
        end
        Z(i,h)=solution(end);
    end
end
end

%% Progress Reporting
function progressReport(message,total)
persistent started completed lastPercent
if isempty(message)
    started=tic; completed=0; lastPercent=-1;
    chainProgress('reset');
    fprintf('Progress: 0%% | Elapsed: 00:00:00 | Remaining: --:--:--\n');
    return;
end
chain=chainProgress('update',message);
isRound=isstruct(message) && isfield(message,'Kind') && strcmp(message.Kind,'round');
if isRound, completed=min(total,completed+message.Count); end
if ~isRound && ~chain.changed, return; end
percent=min(100,floor(100*completed/max(total,1)));
if percent==lastPercent && ~chain.changed, return; end
elapsed=toc(started);
if completed>0
    remaining=formatDuration(elapsed*(total-completed)/completed);
else
    remaining='--:--:--';
end
printChainProgress(percent,elapsed,remaining);
lastPercent=percent;
end

function printChainProgress(percent,elapsed,remaining)
chain=chainProgress('read');
if chain.percent<0
    fprintf('Progress: %d%% | Elapsed: %s | Remaining: %s\n', ...
        percent,formatDuration(elapsed),remaining);
else
    fprintf('Progress: %d%% | Chain: %d%% | Elapsed: %s | Remaining: %s\n', ...
        percent,chain.percent,formatDuration(elapsed),remaining);
end
end

function value=formatDuration(seconds)
seconds=max(0,floor(seconds));
value=sprintf('%02d:%02d:%02d',floor(seconds/3600), ...
    floor(mod(seconds,3600)/60),mod(seconds,60));
end

%% Statistical Summary
function S=summarizeMain(jobs,nD,cfg)
nM=numel(cfg.methods); nR=cfg.numRepeats; T=cfg.rounds;
A=zeros(nD,nR,nM); O=zeros(nD,nR,nM,T); initial=zeros(nD,nR,nM);
seconds=zeros(nD,nR,nM); select=seconds; trajectory=seconds;
fallback=zeros(nD,nR); union=nan(nD,nR,T);
for d=1:nD
    for r=1:nR
        j=(d-1)*nR+r; job=jobs{j};
        A(d,r,:)=reshape(job.AOMAE,1,1,nM);
        O(d,r,:,:)=reshape(job.OMAE,1,1,nM,T);
        initial(d,r,:)=reshape(job.InitialOMAE,1,1,nM);
        seconds(d,r,:)=reshape(job.TotalSeconds,1,1,nM);
        select(d,r,:)=reshape(sum(job.SelectionSeconds,2),1,1,nM);
        trajectory(d,r,:)=reshape(sum(job.TrajectorySeconds,2),1,1,nM);
        fallback(d,r)=sum(job.FallbackUsed(1,:));
        union(d,r,:)=reshape(job.TrajectoryUnionSize(1,:),1,1,T);
    end
end
assert(all(isfinite(A(:))) && all(A(:)>=0&A(:)<=1));
S.Mean=reshape(mean(A,2),nD,nM); S.SD=reshape(std(A,0,2),nD,nM);
S.Overall=mean(S.Mean,1); S.FinalMean=reshape(mean(O(:,:,:,T),2),nD,nM);
S.FinalSD=reshape(std(O(:,:,:,T),0,2),nD,nM);
S.RoundMean=reshape(mean(O,2),nD,nM,T);
S.RoundSD=reshape(std(O,0,2),nD,nM,T);
S.InitialMean=reshape(mean(initial,2),nD,nM);
S.InitialSD=reshape(std(initial,0,2),nD,nM);
% Arithmetic mean of total method runtime over all datasets and repeats.
% Each timed run includes initialization, batch selection, and model updates.
S.MeanSeconds=reshape(mean(seconds,2),nD,nM);
S.OverallMeanSeconds=mean(S.MeanSeconds,1);
S.MedianSeconds=reshape(median(seconds,2),nD,nM);
S.MedianSelectionSeconds=reshape(median(select,2),nD,nM);
S.MedianTrajectorySeconds=reshape(median(trajectory,2),nD,nM);
S.DatasetRanks=zeros(nD,nM);
for d=1:nD, S.DatasetRanks(d,:)=pooledRanks(S.Mean(d,:)')'; end
S.MeanRank=mean(S.DatasetRanks,1);
S.FriedmanQ=12*nD/(nM*(nM+1))*(sum(S.MeanRank.^2)-nM*(nM+1)^2/4);
ties=0;
for d=1:nD
    [~,~,group]=unique(S.Mean(d,:)); count=accumarray(group(:),1);
    ties=ties+sum(count.^3-count);
end
correction=1-ties/(nD*(nM^3-nM));
if correction>0
    S.FriedmanQ=max(0,S.FriedmanQ/correction);
    S.FriedmanP=gammainc(S.FriedmanQ/2,(nM-1)/2,'upper');
else, S.FriedmanQ=0; S.FriedmanP=1; end
% Positive Delta and W/T/L favor AVG--TP--TC. Tests use nine dataset means.
% CIs resample paired repetitions within each fixed dataset.
S.Delta=S.Overall(2:end)-S.Overall(1);
S.RawP=zeros(1,nM-1); S.WTL=zeros(nM-1,3);
for m=2:nM
    difference=S.Mean(:,m)-S.Mean(:,1);
    S.RawP(m-1)=exactSignedRank(difference);
    S.WTL(m-1,:)=[sum(difference>0),sum(difference==0),sum(difference<0)];
end
[sorted,order]=sort(S.RawP); adjusted=zeros(size(sorted)); running=0;
for i=1:numel(sorted)
    running=max(running,(numel(sorted)-i+1)*sorted(i));
    adjusted(i)=min(1,running);
end
S.HolmP=zeros(size(sorted)); S.HolmP(order)=adjusted;
stream=RandStream('mt19937ar','Seed',cfg.bootstrapSeed);
B=zeros(cfg.bootstrapReplicates,nM);
for b=1:cfg.bootstrapReplicates
    for d=1:nD
        ix=randi(stream,nR,nR,1);
        B(b,:)=B(b,:)+mean(reshape(A(d,ix,:),nR,nM),1)/nD;
    end
end
alpha=(1-cfg.confidenceLevel)/2;
S.CI=zeros(nM,2); S.DeltaCI=zeros(nM-1,2);
for m=1:nM
    S.CI(m,:)=[empiricalQuantile(B(:,m),alpha),empiricalQuantile(B(:,m),1-alpha)];
    if m>1
        delta=B(:,m)-B(:,1);
        S.DeltaCI(m-1,:)=[empiricalQuantile(delta,alpha),empiricalQuantile(delta,1-alpha)];
    end
end
S.FallbackFraction=sum(fallback,2)/(nR*T);
S.MeanUnionSize=nan(nD,1);
for d=1:nD
    v=reshape(union(d,:,:),[],1); v=v(isfinite(v));
    if ~isempty(v), S.MeanUnionSize(d)=mean(v); end
end
end

%% Exact Signed-Rank Test
function p=exactSignedRank(difference)
difference=difference(difference~=0); n=numel(difference);
if n==0, p=1; return; end
ranks=pooledRanks(abs(difference)); observed=sum(ranks(difference>0));
center=sum(ranks)/2; extreme=0;
for mask=0:2^n-1
    signs=logical(bitget(uint32(mask),1:n));
    W=sum(ranks(signs));
    extreme=extreme+(abs(W-center)>=abs(observed-center)-1e-12);
end
p=extreme/2^n;
end

%% Result Tables
function writeMainTex(path,S,inputs,cfg)
fid=fopen(path,'w'); assert(fid>=0,'Cannot write %s',path);
closer=onCleanup(@()fclose(fid));
% Three main-text tables and two appendix tables; no table notes.
writeMetricTable(fid,sprintf('AOMAE over %d query rounds on the nine evaluation datasets.',cfg.rounds), ...
    'tab:main_aomae',S.Mean,S.SD,inputs,cfg,true);
writeMetricTable(fid,sprintf('OMAE after %d queries on the nine evaluation datasets.',cfg.rounds*cfg.batchSize), ...
    'tab:main_final',S.FinalMean,S.FinalSD,inputs,cfg,true);
texLine(fid,'\begin{table}[H]');
texLine(fid,'\centering');
texLine(fid,'\caption{Average ranks, paired AOMAE differences, and mean runtime.}');
texLine(fid,'\label{tab:main_statistics}');
texLine(fid,'\fontsize{7.5pt}{10pt}\selectfont');
texLine(fid,'\setlength{\tabcolsep}{4pt}');
texLine(fid,'\begin{tabular}{lcccccr}');
texLine(fid,'\toprule');
texLine(fid,'Method & Mean rank & W/T/L & $\Delta$AOMAE & 95\% CI for $\Delta$ & Holm $p$ & Mean time (s) \\');
texLine(fid,'\midrule');
for m=1:numel(cfg.methods)
    name=displayMethod(cfg.methods{m});
    if m==1
        row=sprintf('%s & %.2f & -- & -- & -- & -- & %.1f', ...
            name,S.MeanRank(m),S.OverallMeanSeconds(m));
    else
        i=m-1;
        row=sprintf('%s & %.2f & %d/%d/%d & %.4f & $[%.4f,%.4f]$ & %.4f & %.1f', ...
            name,S.MeanRank(m),S.WTL(i,:),S.Delta(i), ...
            S.DeltaCI(i,:),S.HolmP(i),S.OverallMeanSeconds(m));
    end
    texLine(fid,[row,' \\']);
end
texLine(fid,'\bottomrule');
texLine(fid,'\end{tabular}');
texLine(fid,'\end{table}');
texLine(fid,'');
for t=1:cfg.rounds-1
    writeMetricTable(fid,sprintf('OMAE after %d queries.',t*cfg.batchSize), ...
        sprintf('tab:main_round_%d',t),S.RoundMean(:,:,t), ...
        S.RoundSD(:,:,t),inputs,cfg,false);
end
[~,writeError]=ferror(fid);
assert(writeError==0,'RQ2:TexWriteFailed','Failed to write the complete TeX file.');
end

function name=displayMethod(method)
switch method
    case 'AVG-TP-TC', name='AVG--TP--TC';
    case 'MS-Raw', name='MS';
    case 'Core-set-VP', name='Core-set';
    otherwise, name=method;
end
end

%% Metric Table
function writeMetricTable(fid,caption,label,mu,sd,inputs,cfg,includeOverall)
texLine(fid,'\begin{table}[H]');
texLine(fid,'\centering');
texLine(fid,['\caption{',caption,'}']);
texLine(fid,['\label{',label,'}']);
texLine(fid,'\fontsize{7pt}{9pt}\selectfont');
texLine(fid,'\setlength{\tabcolsep}{3pt}');
texLine(fid,['\begin{tabular}{l',repmat('c',1,numel(cfg.methods)),'}']);
texLine(fid,'\toprule');
row='Dataset';
for m=1:numel(cfg.methods), row=[row,' & ',displayMethod(cfg.methods{m})]; end
texLine(fid,[row,' \\']);
texLine(fid,'\midrule');
for d=1:numel(inputs)
    row=inputs{d}.Dataset;
    for m=1:numel(cfg.methods)
        value=sprintf('%.4f',mu(d,m));
        if mu(d,m)==min(mu(d,:)), value=['\mathbf{',value,'}']; end
        cellText=['$',value,'\pm ',sprintf('%.4f',sd(d,m)),'$'];
        row=[row,' & ',cellText];
    end
    texLine(fid,[row,' \\']);
end
if includeOverall
    texLine(fid,'\midrule');
    row=['Overall',sprintf(' & %.4f',mean(mu,1))];
    texLine(fid,[row,' \\']);
end
texLine(fid,'\bottomrule');
texLine(fid,'\end{tabular}');
texLine(fid,'\end{table}');
texLine(fid,'');
end

function texLine(fid,line)
% LaTeX is passed as data: backslashes and percent signs remain literal.
fprintf(fid,'%s\n',line);
end

%% Protocol Checks
function protocolTests(cfg)
p=[1,0,0]; assert(adequacyScore(p,2,cfg)==0);
K=[1,0,0;0,1,0;0,0,1];
assert(isequal(greedyCoverage(K,[0.2;0.6;0.2],2,[1;2;3]),[2;1]));
assert(abs(exactSignedRank(ones(9,1))-2/512)<1e-12);
assert(exactSignedRank(zeros(9,1))==1);
R=struct('p',1,'q',2,'nVar',2,'y',zeros(0,1),'epsilon',cfg.epsilonAssignment, ...
    'lpA',[0,-1;0,1],'lpRhs',[0;1],'lpEq',[1,0],'lpLb',[-inf;-inf]);
z=classMaximumMargins(R,[0.2;0.8],cfg);
expected=[0.8,0.2-cfg.epsilonAssignment;0.2,0.8-cfg.epsilonAssignment];
assert(max(abs(z(:)-expected(:)))<1e-7,'Class maximum-margin LP test failed.');
z=[0,0,0;4,0,0];
for method={'MS-Raw','ES','LS'}
    values=liAcquisitionScore(z,method{1}); assert(values(1)>values(2));
    shifted=liAcquisitionScore(z+1000,method{1});
    assert(max(abs(values-shifted))<1e-10);
end
assert(isequal(liAcquisitionScore(z,'SM'),[0;4]));
Rt=R; Rt.S=[-1,0;0,-1;0,1]; Rt.sb=[0;0;1]; Rt.eq=[1,0];
Rt.Phi=zeros(0,1);
zLi=liHypotheticalObjectives(Rt,[0.2;0.8],cfg);
assert(all(isfinite(zLi(:))) && zLi(1,1)>zLi(2,1) && abs(zLi(1,2)-zLi(2,2))<1e-7);
Rt.Phi=[0.2;0.8]; Rt.y=[1;2];
zLi=liHypotheticalObjectives(Rt,[0.2;0.8],cfg);
assert(zLi(1,1)>zLi(1,2) && zLi(2,2)>zLi(2,1), ...
    'Hypothetical-label reoptimization test failed.');
chosen=codeCoreSet(0,[0.25;0.5;0.75;1],2,(1:4)',cfg);
assert(isequal(chosen,[4;2]),'Code Core-set sequential-update test failed.');
chosen=codeCoreSet(0,[1;1;0],3,[10;20;30],cfg);
assert(isequal(chosen,[10;20;30]),'Code Core-set duplicate/tie test failed.');
chosen=codeCoreSet(0,[1;1+cfg.coreTieTol/8],1,[10;20],cfg);
assert(isequal(chosen,10),'Code Core-set tolerance tie test failed.');
end

%% Model State
function [base,order,newFit]=getState(ref,labels,context,cfg,cache,order,isRealState,distanceKind)
[ref,ix]=sort(ref); labels=labels(ix);
assert(numel(unique(ref))==numel(ref),'Duplicate reference alternative.');
key=sprintf('%d:%d;',reshape([ref(:),labels(:)]',[],1));
if isKey(cache,key)
    base=cache(key); newFit=false;
    if isRealState && ~base.IsRealState
        base.IsRealState=true; cache(key)=base;
        modelCachePromote(context.ModelCache,key);
    end
    position=find(strcmp(order,key),1); order(position)=[]; order{end+1}=key;
    base=ensureEvidence(base,distanceKind,context,cfg); cache(key)=base;
    return;
end
[modelEntry,modelHit]=modelCacheGet(context.ModelCache,key);
newFit=~modelHit;
if modelHit
    R=modelEntry.Region; M=modelEntry.Models; seeds=modelEntry.Seeds;
    lp=modelEntry.LPInfo; geometry=modelEntry.Geometry; audits=modelEntry.Audit;
    if isRealState && (~isfield(modelEntry,'IsRealState') || ~modelEntry.IsRealState)
        modelCachePromote(context.ModelCache,key);
    end
else
    seed=keySeed(cfg.chainSeed,context.DatasetID,context.Repeat,key);
    [R,lp]=makeRegion(context.X(ref,:),labels,context.q,cfg);
    seeds=zeros(1,cfg.nChains);
    for ch=1:cfg.nChains, seeds(ch)=keySeed(seed,0,ch,'chain'); end
    [starts,~]=makeStarts(R,seeds,cfg);
    [R,geometry]=preparePreconditioner(R,starts,seeds,cfg);
    M=zeros(R.nVar,cfg.Ns); audits=cell(cfg.nChains,1);
    perChain=cfg.Ns/cfg.nChains;
    for ch=1:cfg.nChains
        samplingCfg=cfg; samplingCfg.queue=cfg.progressQueue;
        samplingCfg.progressContext=struct('Kind','transition','DatasetID',context.DatasetID, ...
            'Repeat',context.Repeat,'Chain',ch);
        rng(seeds(ch),'twister'); [chain,audit]=sampleChain(R,starts(:,ch),samplingCfg);
        assert(audit.NumericalFailures==0,'Development:NumericalFailure', ...
            'Sampler numerical failures; inspect geometry. No chain shortening or resampling.');
        M(:,(ch-1)*perChain+(1:perChain))=chain';
        audits{ch}=audit;
    end
    modelEntry=struct('Region',R,'Models',M,'Seeds',seeds,'LPInfo',lp, ...
        'Geometry',geometry,'Audit',{audits},'IsRealState',isRealState);
    modelCachePut(context.ModelCache,key,modelEntry,cfg.modelCacheMB*1024^2);
end
ids=[context.Universe;ref]; allP=context.P(ids,:); U=allP*M(1:R.p,:);
assignment=ones(size(U));
for h=1:R.q-1, assignment=assignment+(U>M(R.p+h,:)); end
support=zeros(numel(ids),R.q);
for h=1:R.q, support(:,h)=mean(assignment==h,2); end
needRAD=~isfield(cfg,'progressMethod') || strcmp(cfg.progressMethod,'AVG-TP-TC');
rad=[];
if needRAD
    robust=zeros(numel(ids),R.q);
    for h=1:R.q
        rho=classDeviation(U,M,R,h); robust(:,h)=max(rho,[],2);
    end
    rad=min(robust,[],2);
end
base=struct('Region',R,'Models',M,'IDs',ids,'U',U,'Assignment',assignment, ...
    'MS',support,'TP',[],'RAD',rad, ...
    'ReferenceIDs',ref,'Labels',labels,'Seeds',seeds,'StateKey',key, ...
    'LPInfo',lp,'Geometry',geometry,'Audit',{audits},'IsRealState',isRealState);
base=ensureEvidence(base,distanceKind,context,cfg);
if cache.Count>=cfg.cacheStates
    victim=1;
    for j=1:numel(order)
        entry=cache(order{j});
        if ~entry.IsRealState, victim=j; break; end
    end
    remove(cache,order{victim}); order(victim)=[];
end
cache(key)=base; order{end+1}=key;
end

%% Evidence Construction
function base=ensureEvidence(base,kind,context,cfg)
if strcmp(kind,'none'), return; end
assert(strcmp(kind,'TP'),'This benchmark uses only TP local evidence.');
if ~isempty(base.TP), return; end
nPool=numel(context.Universe);
activeRows=find(~ismember(context.Universe,base.ReferenceIDs));
D=thresholdDistance(base.U,base.U(nPool+1:end,:),base.Models,base.Region,cfg);
base.TP=integratedEvidence(base.MS,D,base.Labels,cfg,activeRows);
end

%% Model Cache Get
function [entry,hit]=modelCacheGet(cache,key)
hit=isKey(cache,key); entry=[];
if ~hit, return; end
entry=cache(key); info=cache('__meta__');
position=find(strcmp(info.Order,key),1);
info.Order(position)=[]; info.Order{end+1}=key; info.Hits=info.Hits+1;
cache('__meta__')=info;
end

%% Model Cache Put
function modelCachePut(cache,key,entry,budget)
if budget<=0, return; end
measurement=whos('entry'); entry.CacheBytes=measurement.bytes+64;
if entry.CacheBytes>budget, return; end
info=cache('__meta__');
while ~isempty(info.Order) && info.Bytes+entry.CacheBytes>budget
    victimIndex=modelCacheVictim(cache,info.Order);
    victim=info.Order{victimIndex}; previous=cache(victim);
    info.Bytes=info.Bytes-previous.CacheBytes; remove(cache,victim); info.Order(victimIndex)=[];
end
cache(key)=entry; info.Order{end+1}=key; info.Bytes=info.Bytes+entry.CacheBytes;
cache('__meta__')=info;
end

%% Model Cache Victim
function victimIndex=modelCacheVictim(cache,order)
victimIndex=1;
for j=1:numel(order)
    entry=cache(order{j});
    if ~isfield(entry,'IsRealState') || ~entry.IsRealState
        victimIndex=j; return;
    end
end
end

%% Model Cache Promote
function modelCachePromote(cache,key)
if ~isKey(cache,key), return; end
entry=cache(key);
if isfield(entry,'IsRealState') && entry.IsRealState, return; end
entry.IsRealState=true;
cache(key)=entry;
end

%% Adequacy Score
function score=adequacyScore(p,hat,cfg)
q=size(p,2); distance=abs(hat(:)-(1:q));
if strcmp(cfg.adequacyNormalization,'recommendation_max')
    denominator=max(hat(:)-1,q-hat(:));
else, error('Use the revised recommendation_max normalization.'); end
score=1-sum(p.*distance,2)./denominator;
assert(all(score>=-1e-12&score<=1+1e-12)); score=max(0,min(1,score));
end

%% Assign Classes
function labels=assignClasses(U,b,q)
labels=ones(size(U));
for h=1:q-1, labels=labels+(U>b(h)); end
end

%% Class Deviation
function rho=classDeviation(U,M,R,h)
if h==1, lower=zeros(size(U)); else, lower=M(R.p+h-1,:)+R.epsilon-U; end
if h==R.q, upper=zeros(size(U)); else, upper=U-M(R.p+h,:); end
rho=max(0,max(lower,upper));
end

%% Threshold Distance
function D=thresholdDistance(Ua,Ub,M,R,cfg)
D=zeros(size(Ua,1),size(Ub,1));
if isequal(Ua,Ub)
    for h=1:R.q
        z=classDeviation(Ua,M,R,h);
        for j=1:size(Ua,1)-1
            D(j+1:end,j)=D(j+1:end,j)+mean(abs(z(j+1:end,:)-z(j,:)),2);
        end
    end
    D=D+D';
else
    for h=1:R.q
        za=classDeviation(Ua,M,R,h); zb=classDeviation(Ub,M,R,h);
        for j=1:size(Ub,1), D(:,j)=D(:,j)+mean(abs(za-zb(j,:)),2); end
    end
end
assert(all(isfinite(D(:)))&&all(D(:)>=0));
end

%% Integrated Evidence
function p=integratedEvidence(ms,D,labels,cfg,activeRows)
sigma=distanceBandwidth(D(activeRows,:),cfg);
K=exp(-(D/sigma).^2); W=sum(K,2); local=zeros(size(ms));
for h=1:size(ms,2), local(:,h)=sum(K(:,labels==h),2); end
p=(ms+local)./(1+W);
assert(max(abs(sum(p,2)-1))<1e-10);
end

%% Distance Bandwidth
function sigma=distanceBandwidth(D,cfg)
positive=D(D>0);
if isempty(positive), sigma=eps; else, sigma=cfg.etaSigma*median(positive); end
end

%% Normalized Weights
function w=normalizedWeights(v)
if sum(v)>0, w=v/sum(v); else, w=ones(size(v))/numel(v); end
end

%% Greedy Coverage
function chosen=greedyCoverage(K,weights,k,ids)
assert(size(K,1)>=k&&all(K(:)>=0)&&all(weights>=0));
[ids,order]=sort(ids); K=K(order,:); weights=weights(:);
covered=zeros(1,size(K,2)); available=true(numel(ids),1); chosen=zeros(k,1);
for step=1:k
    gains=max(0,K-covered)*weights; gains(~available)=-inf;
    [~,row]=max(gains); chosen(step)=ids(row); available(row)=false;
    covered=max(covered,K(row,:));
end
end

%% Key Seed
function seed=keySeed(base,d,r,key)
prime=2147483647;
seed=mod(double(base)+104729*d+13007*r,prime);
bytes=double(char(key));
for j=1:numel(bytes), seed=mod(seed*131+bytes(j),prime); end
seed=max(1,floor(seed));
end

%% Read Data
function [X,y,info]=readData(file)
T=readtable(file,'VariableNamingRule','preserve'); names=string(T.Properties.VariableNames);
g=find(startsWith(names,'g','IgnoreCase',true)); h=find(strcmpi(names,'Class'));
assert(~isempty(g)&&numel(h)==1,'Expected g* columns and one Class column.');
X=double(table2array(T(:,g))); labels=string(T{:,h}); y=zeros(numel(labels),1);
for i=1:numel(y)
    token=regexp(strtrim(char(labels(i))),'^(?:(?:Cl)|(?:C))?([0-9]+)$','tokens','once');
    assert(~isempty(token),'Invalid ordered label: %s',labels(i)); y(i)=str2double(token{1});
end
assert(all(isfinite(X(:)))&&isequal(unique(y),(1:max(y))'),'Missing/nonfinite data or nonconsecutive classes.');
assert(max(y)>=2,'At least two classes required.');
lo=min(X,[],1); hi=max(X,[],1); transformed=false(1,size(X,2));
for j=1:size(X,2)
    if lo(j)<-1e-10||hi(j)>1+1e-10
        if hi(j)>lo(j), X(:,j)=(X(:,j)-lo(j))/(hi(j)-lo(j)); else, X(:,j)=0; end
        transformed(j)=true;
    end
end
info=struct('Minimum',lo,'Maximum',hi,'Transformed',transformed, ...
    'FeatureRangeSource','All supplied unlabeled feature rows; matches existing offline preprocessing');
end

%% Split Data
function [ref,pool,test]=splitData(y,nref,npool)
q=max(y); counts=accumarray(y,1,[q,1]); rem=counts-nref;
assert(all(rem>=2)&&npool>=q&&sum(rem)-npool>=q,'Insufficient class counts.');
target=npool*rem/sum(rem); alloc=min(rem-1,max(1,floor(target)));
while sum(alloc)<npool
    score=target-alloc; score(alloc>=rem-1)=-inf; [~,j]=max(score); alloc(j)=alloc(j)+1;
end
while sum(alloc)>npool
    score=alloc-target; score(alloc<=1)=-inf; [~,j]=max(score); alloc(j)=alloc(j)-1;
end
ref=[];pool=[];test=[];
for h=1:q
    ids=find(y==h); ids=ids(randperm(numel(ids)));
    ref=[ref;ids(1:nref)]; pool=[pool;ids(nref+(1:alloc(h)))];
    test=[test;ids(nref+alloc(h)+1:end)];
end
ref=ref(randperm(numel(ref))); pool=pool(randperm(numel(pool))); test=test(randperm(numel(test)));
end

%% Basis
function P=basis(X,L)
P=zeros(size(X,1),size(X,2)*L);
for j=1:size(X,2)
    for k=1:L, P(:,(j-1)*L+k)=max(0,min(1,L*X(:,j)-(k-1))); end
end
end

%% Make Region
function [R,info]=makeRegion(X,y,q,cfg)
P=basis(X,cfg.L); p=size(P,2); n=numel(y); nv=p+q-1;
eq=[ones(1,p),zeros(1,q-1)]; B=null(eq);
S=[-eye(p),zeros(p,q-1)]; sb=zeros(p,1);
row=zeros(1,nv); row(p+1)=-1; S=[S;row]; sb=[sb;0];
for h=2:q-1
    row=zeros(1,nv); row(p+h-1)=1; row(p+h)=-1; S=[S;row]; sb=[sb;0];
end
row=zeros(1,nv); row(end)=1; S=[S;row]; sb=[sb;1];
lossRows=[]; lossRhs=[]; owner=[];
for i=1:n
    if y(i)>1
        row=[-P(i,:),zeros(1,q-1)]; row(p+y(i)-1)=1;
        lossRows=[lossRows;row]; lossRhs=[lossRhs;-cfg.epsilonAssignment]; owner=[owner;i];
    end
    if y(i)<q
        row=[P(i,:),zeros(1,q-1)]; row(p+y(i))=-1;
        lossRows=[lossRows;row]; lossRhs=[lossRhs;0]; owner=[owner;i];
    end
end
E=sparse((1:numel(owner))',owner,-1,numel(owner),n);
A=[S,zeros(size(S,1),n);lossRows,full(E)]; rhs=[sb;lossRhs];
Aeq=[eq,zeros(1,n)]; lb=[-inf(nv,1);zeros(n,1)];
f=[zeros(nv,1);ones(n,1)];
[sol,fval,solver]=solveLP(f,A,rhs,Aeq,1,lb,[],cfg);
R=struct('p',p,'q',q,'nVar',nv,'B',B,'S',S,'sb',sb,'Phi',P, ...
    'y',y,'epsilon',cfg.epsilonAssignment,'eq',eq,'eqValue',1);
actual=lossValue(sol(1:nv),R);
R.phiStar=max([0,fval,actual]); R.cap=R.phiStar+cfg.epsilonPhi;
capRow=[zeros(1,nv),ones(1,n)]; A=[A;capRow]; rhs=[rhs;R.cap];
norms=[sqrt(sum((S*B).^2,2));sqrt(sum((lossRows*B).^2,2));0];
ballA=[A,norms]; ballEq=[Aeq,0]; ballF=[zeros(nv+n,1);-1];
[ball,~,centerSolver]=solveLP(ballF,ballA,rhs,ballEq,1,[lb;0],[],cfg);
R.center=ball(1:nv); R.radius=ball(end);
assert(R.radius>cfg.minimumRadius,'HAR:Degenerate', ...
    'No certified full-relative-dimensional ball. Do not increase epsilon silently; inspect LP/affine hull.');
assertFeasible(R.center,R,cfg);
R.lpA=A; R.lpRhs=rhs; R.lpEq=Aeq; R.lpLb=lb;
info=struct('MinimumLossObjective',fval,'ActualLossAtOptimum',actual, ...
    'NumericalLossGap',actual-fval,'MinimumLossSolver',solver,'CenterSolver',centerSolver, ...
    'BallConstruction','Conservative sum of per-object worst-branch bounds');
end

%% Solve Lp
function [x,fval,report]=solveLP(f,A,b,Aeq,beq,lb,ub,cfg)
persistent defaultOptions optionsTolerance
if isempty(defaultOptions) || ~isequal(optionsTolerance,cfg.solverTolerance)
    defaultOptions=optimoptions('linprog','Display','none', ...
        'ConstraintTolerance',cfg.solverTolerance,'OptimalityTolerance',cfg.solverTolerance);
    optionsTolerance=cfg.solverTolerance;
end
opts=defaultOptions;
As=sparse(A); Es=sparse(Aeq);
[x,fval,flag,output]=linprog(f,As,b,Es,beq,lb,ub,opts);
res=lpPrimalResidual(x,As,b,Es,beq,lb,ub);
accepted=lpSolutionAccepted(x,fval,flag,res,cfg);
attempts={struct('Algorithm',opts.Algorithm,'ExitFlag',flag, ...
    'MaxPrimalResidual',res,'Output',output)};
if accepted
    report=lpReport(flag,output,res,opts.Algorithm,attempts);
    return;
end
originalRng=rng; rngCleanup=onCleanup(@()rng(originalRng));
warning('Comparison:LPRetry', ...
    ['LP failed validation (exitflag=%d, residual=%.3g). Retrying the same LP ', ...
     'with MaxIterations=%d. Original message: %s'], ...
    flag,res,cfg.lpRetryMaxIterations,output.message);
retry=optimoptions(opts,'MaxIterations',cfg.lpRetryMaxIterations);
[x,fval,flag,output]=linprog(f,As,b,Es,beq,lb,ub,retry);
res=lpPrimalResidual(x,As,b,Es,beq,lb,ub);
attempts{end+1}=struct('Algorithm',retry.Algorithm,'ExitFlag',flag, ...
    'MaxPrimalResidual',res,'Output',output);
if lpSolutionAccepted(x,fval,flag,res,cfg)
    report=lpReport(flag,output,res,retry.Algorithm,attempts);
    return;
end
warning('Comparison:LPFallback', ...
    ['LP retry failed validation (exitflag=%d, residual=%.3g). ', ...
     'Trying %s with the same objective, constraints and tolerances.'], ...
    flag,res,cfg.lpFallbackAlgorithm);
fallback=optimoptions('linprog','Display','none', ...
    'Algorithm',cfg.lpFallbackAlgorithm,'MaxIterations',cfg.lpFallbackMaxIterations, ...
    'ConstraintTolerance',cfg.solverTolerance,'OptimalityTolerance',cfg.solverTolerance);
[x,fval,flag,output]=linprog(f,As,b,Es,beq,lb,ub,fallback);
res=lpPrimalResidual(x,As,b,Es,beq,lb,ub);
attempts{end+1}=struct('Algorithm',fallback.Algorithm,'ExitFlag',flag, ...
    'MaxPrimalResidual',res,'Output',output);
if ~lpSolutionAccepted(x,fval,flag,res,cfg)
    error('HAR:LP', ...
        ['LP failed after default solver, increased-iteration retry and %s fallback. ', ...
         'Final exitflag=%d; original-constraint residual=%.3g (limit %.3g). %s'], ...
        fallback.Algorithm,flag,res,cfg.feasibilityTolerance,output.message);
end
report=lpReport(flag,output,res,fallback.Algorithm,attempts);
end

%% Lp Primal Residual
function res=lpPrimalResidual(x,A,b,Aeq,beq,lb,ub)
res=inf;
if isempty(x) || ~all(isfinite(x)), return; end
v=0;
if ~isempty(A), v=[v;A*x-b]; end
if ~isempty(Aeq), v=[v;abs(Aeq*x-beq)]; end
if ~isempty(lb), v=[v;lb-x]; end
if ~isempty(ub), v=[v;x-ub]; end
if any(isnan(v)), return; end
res=max(v);
end

%% Lp Solution Accepted
function ok=lpSolutionAccepted(x,fval,flag,res,cfg)
ok=flag>0 && ~isempty(x) && all(isfinite(x)) && ...
    isscalar(fval) && isfinite(fval) && isfinite(res) && res<=cfg.feasibilityTolerance;
end

%% Lp Report
function report=lpReport(flag,output,res,algorithm,attempts)
report=struct('ExitFlag',flag,'Message',output.message, ...
    'MaxPrimalResidual',res,'Algorithm',algorithm, ...
    'Retried',numel(attempts)>1,'Attempts',{attempts});
end

%% Make Starts
function [starts,info]=makeStarts(R,seeds,cfg)
starts=zeros(R.nVar,numel(seeds)); info=struct([]);
for c=1:numel(seeds)
    rng(seeds(c)+17,'twister'); objective=[R.B*randn(size(R.B,2),1);zeros(numel(R.y),1)];
    [sol,~,report]=solveLP(objective,R.lpA,R.lpRhs,R.lpEq,1,R.lpLb,[],cfg);
    starts(:,c)=0.7*R.center+0.3*sol(1:R.nVar);
    assertFeasible(starts(:,c),R,cfg);
    report.DistanceFromCenter=norm(starts(:,c)-R.center); info=appendStruct(info,report);
end
end

%% Prepare Preconditioner
function [R,report]=preparePreconditioner(R,starts,seeds,cfg)
assert(strcmp(cfg.preconditioningMethod,'hessian'),'Only fixed Hessian geometry is supported.');
if cfg.usePreconditioning
    [R,report]=prepareHessianGeometry(R,cfg);
else
    R.DirectionBasis=R.B; report=table();
end
end

%% Prepare Hessian Geometry
function [R,report]=prepareHessianGeometry(R,cfg)
t0=tic; x0=R.center; U=R.Phi*x0(1:R.p); b=x0(R.p+1:end);
lower=zeros(numel(R.y),1); upper=lower;
ids=R.y>1; lower(ids)=b(R.y(ids)-1)+R.epsilon-U(ids);
ids=R.y<R.q; upper(ids)=U(ids)-b(R.y(ids));
individualLoss=max(0,max(lower,upper));
remaining=R.cap-sum(individualLoss);
assert(remaining>0,'HAR:InnerGeometry','Center has no strict loss-cap slack.');
allowance=individualLoss+0.5*remaining/numel(R.y);
A=R.S; rhs=R.sb;
for i=1:numel(R.y)
    h=R.y(i);
    if h>1
        row=[-R.Phi(i,:),zeros(1,R.q-1)]; row(R.p+h-1)=1;
        A=[A;row]; rhs=[rhs;allowance(i)-R.epsilon];
    end
    if h<R.q
        row=[R.Phi(i,:),zeros(1,R.q-1)]; row(R.p+h)=-1;
        A=[A;row]; rhs=[rhs;allowance(i)];
    end
end
Ar=A*R.B; br=rhs-A*x0; norms=sqrt(sum(Ar.^2,2)); keep=norms>0;
assert(all(br(~keep)>0),'HAR:InnerZeroRow','Nonpositive constant inner constraint.');
Ar=Ar(keep,:)./norms(keep); br=br(keep)./norms(keep);
assert(all(br>0),'HAR:InnerStart','Geometry center must be strictly interior.');
z=zeros(size(R.B,2),1); converged=false; decrement=NaN; iterations=0;
for iter=1:cfg.barrierMaxIterations
    iterations=iter; slack=br-Ar*z; W=Ar./slack;
    gradient=Ar'*(1./slack); H=W'*W;
    if rcond(H)<1e-12
        H=H+1e-12*max(1,trace(H)/size(H,1))*eye(size(H));
    end
    step=-H\gradient; decrement=-gradient'*step;
    assert(all(isfinite(step))&&isfinite(decrement),'HAR:Barrier','Nonfinite Newton step.');
    if decrement>=0&&decrement<cfg.barrierNewtonTolerance
        converged=true; break;
    end
    assert(decrement>=0,'HAR:BarrierDirection','Newton direction is not a descent direction.');
    fraction=1; directional=Ar*step; positive=directional>0;
    if any(positive)
        fraction=min(fraction,cfg.barrierStepFraction*min(slack(positive)./directional(positive)));
    end
    old=-sum(log(slack)); accepted=false;
    for bt=1:60
        candidate=z+fraction*step; newSlack=br-Ar*candidate;
        if all(newSlack>0)&&-sum(log(newSlack))<=old+cfg.barrierArmijo*fraction*(gradient'*step)
            z=candidate; accepted=true; break;
        end
        fraction=fraction/2;
    end
    if ~accepted, break; end
end
center=x0+R.B*z; assertFeasible(center,R,cfg);
slack=br-Ar*z; W=Ar./slack;
[~,factor]=qr(W,0); regularized=false;
if rcond(factor)<1e-10
    [~,singular,V]=svd(W,0); sv=diag(singular);
    assert(max(sv)>0,'HAR:BarrierRank','Invalid barrier geometry.');
    factor=diag(max(sv,1e-6*max(sv)))*V'; regularized=true;
end
R.DirectionBasis=R.B/factor;
R.DirectionBasis=R.DirectionBasis/norm(R.DirectionBasis,'fro');
R.GeometryCenter=center;
report=table("hessian",toc(t0),iterations,converged,decrement, ...
    cond(factor),regularized,remaining, ...
    'VariableNames',{'Method','Seconds','Iterations','CenterConverged', ...
    'NewtonDecrement','FactorCondition','Regularized','InitialLossSlack'});
if cfg.displayDetails
    fprintf('=== Fixed analytic-center/Hessian geometry ===\n'); disp(report);
end
if cfg.verbose
fprintf('Geometry-only inner polytope; full original loss region is used for every transition.\n');
end
if ~converged && cfg.verbose
    fprintf('Geometry center solve stopped early; retaining the verified interior point.\n');
end
end

%% Sample Chain
function [chain,audit]=sampleChain(R,start,cfg)
t0=tic; x=start;
burnIn=cfg.burnIn; chainDraws=cfg.chainDraws; nTrans=burnIn+chainDraws;
keepThin=chainDraws/(cfg.Ns/cfg.nChains);
chain=zeros(cfg.Ns/cfg.nChains,R.nVar); kept=0;
minWidth=inf; maxWidth=0;
failures=0; minStep=inf; maxResidual=0; maxEq=0; maxLoss=0;
[sr,er,lr]=residuals(x,R);
sinceReport=0;
if isfield(R,'DirectionBasis'), directionBasis=R.DirectionBasis; else, directionBasis=R.B; end
useCoordinates=strcmp(cfg.directionStrategy,'coordinate_mixture');
sphereProbability=cfg.sphereProbability;
nDir=size(directionBasis,2);
coordinateDirections=directionBasis;
for j=1:nDir
    coordinateDirections(:,j)=directionBasis(:,j)/norm(directionBasis(:,j));
end
S=R.S; sb=R.sb; Phi=R.Phi; yy=R.y; p=R.p; qq=R.q;
cap=R.cap; epsilon=R.epsilon; eqRow=R.eq; eqValue=R.eqValue;
feasTol=cfg.feasibilityTolerance; lineIterations=cfg.lineIterations;
useMex=isfield(cfg,'useMexSampler') && cfg.useMexSampler;
if useMex && exist('har_chord_mex','file')~=3
    error('Development:MissingMex','cfg.useMexSampler is true but har_chord_mex is not compiled.');
end
n=numel(yy); yGt1=yy>1; yLtQ=yy<qq; yLo=yy(yGt1)-1; yUp=yy(yLtQ);
lo=zeros(n,1); up=zeros(n,1); lod=zeros(n,1); upd=zeros(n,1);
lower=zeros(n,1); upper=zeros(n,1);
hasQueue=isfield(cfg,'queue') && ~isempty(cfg.queue);
hasContext=hasQueue && isfield(cfg,'progressContext');
progressEvery=cfg.progressEvery;
verboseSteps=cfg.verbose && chainDraws>=10000;
reportClock=tic;
for t=1:nTrans
    if useCoordinates && (sphereProbability==0 || rand>sphereProbability)
        j=randi(nDir); d=coordinateDirections(:,j);
    else
        v=randn(nDir,1); d=directionBasis*v; d=d/norm(d);
    end
    if useMex
        [left,right]=har_chord_mex(x,d,S,sb,Phi,yy,p,qq,epsilon,cap,feasTol,lineIterations);
    else
        Sx=S*x; rawSlack=sb-Sx;
        if ~all(rawSlack>=-feasTol)
            error('HAR:Outside','Start outside structure.');
        end
        slack=max(rawSlack,0); Sd=S*d; left=-inf; right=inf;
        positive=Sd>0; negative=Sd<0;
        if any(positive), right=min(slack(positive)./Sd(positive)); end
        if any(negative), left=max(slack(negative)./Sd(negative)); end
        if ~(isfinite(left)&&isfinite(right)&&left<=0&&right>=0)
            error('HAR:Chord','Unbounded/invalid structural chord.');
        end
        if n>0
            Ux=Phi*x(1:p); Ud=Phi*d(1:p); bx=x(p+1:end); bd=d(p+1:end);
            lo(:)=0; up(:)=0; lod(:)=0; upd(:)=0;
            lo(yGt1)=bx(yLo)+epsilon-Ux(yGt1); lod(yGt1)=bd(yLo)-Ud(yGt1);
            up(yLtQ)=Ux(yLtQ)-bx(yUp); upd(yLtQ)=Ud(yLtQ)-bd(yUp);
            originalLeft=left; originalRight=right;
            if sum(max(0,max(lo+left*lod,up+left*upd)))>cap
                left=affineEndpoint(left,lo,up,lod,upd,cap,lineIterations);
            end
            if sum(max(0,max(lo+right*lod,up+right*upd)))>cap
                right=affineEndpoint(right,lo,up,lod,upd,cap,lineIterations);
            end
            xL=x+left*d; UL=Phi*xL(1:p); bL=xL(p+1:end);
            lower(:)=0; upper(:)=0;
            lower(yGt1)=bL(yLo)+epsilon-UL(yGt1); upper(yLtQ)=UL(yLtQ)-bL(yUp);
            if sum(max(0,max(lower,upper)))>cap
                left=bisectEndpoint(originalLeft,x,d,R,cfg);
            end
            xR=x+right*d; UR=Phi*xR(1:p); bR=xR(p+1:end);
            lower(:)=0; upper(:)=0;
            lower(yGt1)=bR(yLo)+epsilon-UR(yGt1); upper(yLtQ)=UR(yLtQ)-bR(yUp);
            if sum(max(0,max(lower,upper)))>cap
                right=bisectEndpoint(originalRight,x,d,R,cfg);
            end
        end
    end
    width=right-left;
    if ~(isfinite(width)&&width>=0)
        error('HAR:Chord','Invalid chord.');
    end
    minWidth=min(minWidth,width); maxWidth=max(maxWidth,width);
    if width<=64*eps(max(1,norm(x)))
        failures=failures+1;
    else
        u=rand; candidate=x+(left+u*width)*d;
        srCandidate=max([0;S*candidate-sb]);
        erCandidate=max([0;abs(eqRow*candidate-eqValue)]);
        if n==0
            lossCandidate=0;
        else
            Uc=Phi*candidate(1:p); bc=candidate(p+1:end);
            lower(:)=0; upper(:)=0;
            lower(yGt1)=bc(yLo)+epsilon-Uc(yGt1); upper(yLtQ)=Uc(yLtQ)-bc(yUp);
            lossCandidate=sum(max(0,max(lower,upper)));
        end
        lrCandidate=max(0,lossCandidate-cap);
        if srCandidate>feasTol || erCandidate>feasTol || lrCandidate>0 || ~all(isfinite(candidate))
            failures=failures+1;
        else
            minStep=min(minStep,norm(candidate-x)); x=candidate;
            sr=srCandidate; er=erCandidate; lr=lrCandidate;
        end
    end
    maxResidual=max(maxResidual,sr); maxEq=max(maxEq,er); maxLoss=max(maxLoss,lr);
    if t>burnIn && mod(t-burnIn,keepThin)==0
        kept=kept+1; chain(kept,:)=x';
    end
    if hasQueue
        sinceReport=sinceReport+1;
        if (mod(t,progressEvery)==0 && toc(reportClock)>=15) || t==nTrans
            if hasContext
                packet=cfg.progressContext; packet.Count=sinceReport; packet.Transition=t; packet.TotalTransitions=nTrans; send(cfg.queue,packet);
            else
                send(cfg.queue,sinceReport);
            end
            sinceReport=0; reportClock=tic;
        end
    end
    if verboseSteps && mod(t,10000)==0
        fprintf('  transitions %d/%d\n',t,nTrans);
    end
end
audit=struct('Seconds',toc(t0),'Transitions',nTrans,'BurnIn',burnIn, ...
    'ContinuousDraws',chainDraws,'StoredDraws',kept,'StorageThin',keepThin,'NumericalFailures',failures,'MinChord',minWidth, ...
    'MaxChord',maxWidth,'MinStep',minStep,'MaxStructuralResidual',maxResidual, ...
    'MaxEqualityResidual',maxEq,'MaxLossResidual',maxLoss);
end

%% Line Loss
function value=lineLoss(t,lo,up,lod,upd)
value=sum(max(0,max(lo+t*lod,up+t*upd)));
end

%% Affine Endpoint
function endpoint=affineEndpoint(outside,lo,up,lod,upd,cap,iterations)
original=outside; orientation=sign(outside); t=abs(outside);
l=orientation*lod; u=orientation*upd;
for step=1:2*numel(lo)+2
    low=lo+t*l; high=up+t*u; value=sum(max(0,max(low,high)));
    if value<=cap, endpoint=orientation*t; return; end
    lowerActive=low>=high & low>0; upperActive=high>low & high>0;
    slope=sum(l(lowerActive))+sum(u(upperActive));
    if ~isfinite(slope)||slope<=0, break; end
    candidate=t-(value-cap)/slope;
    if ~isfinite(candidate)||candidate<0||candidate>=t, break; end
    t=candidate;
end
endpoint=bisectLine(original,lo,up,lod,upd,cap,iterations);
end

%% Bisect Line
function endpoint=bisectLine(outside,lo,up,lod,upd,cap,iterations)
inside=0;
for k=1:iterations
    middle=0.5*(outside+inside);
    if middle==outside||middle==inside, break; end
    if lineLoss(middle,lo,up,lod,upd)<=cap
        inside=middle;
    else
        outside=middle;
    end
end
endpoint=inside;
end

%% Bisect Endpoint
function endpoint=bisectEndpoint(outside,x,d,R,cfg)
inside=0;
assert(lossValue(x,R)<=R.cap,'HAR:LossStart','Current point exceeds exact loss cap.');
for k=1:cfg.lineIterations
    middle=0.5*(outside+inside);
    if middle==outside||middle==inside, break; end
    if lossValue(x+middle*d,R)<=R.cap, inside=middle; else, outside=middle; end
end
endpoint=inside;
end

%% Loss Value
function value=lossValue(x,R)
if isempty(R.y), value=0; return; end
U=R.Phi*x(1:R.p); b=x(R.p+1:end); lower=zeros(numel(R.y),1); upper=lower;
ids=R.y>1; lower(ids)=b(R.y(ids)-1)+R.epsilon-U(ids);
ids=R.y<R.q; upper(ids)=U(ids)-b(R.y(ids));
value=sum(max(0,max(lower,upper)));
end

%% Residuals
function [s,e,l]=residuals(x,R)
s=max([0;R.S*x-R.sb]); e=max([0;abs(R.eq*x-R.eqValue)]); l=max(0,lossValue(x,R)-R.cap);
end

%% Assert Feasible
function assertFeasible(x,R,cfg)
[s,e,l]=residuals(x,R);
assert(all(isfinite(x))&&max([s,e,l])<=cfg.feasibilityTolerance, ...
    'HAR:Feasibility','Invalid parameter: structural=%g equality=%g loss=%g',s,e,l);
end

%% Pooled Ranks
function ranks=pooledRanks(x)
[v,order]=sort(x(:));
first=[1;find(diff(v)~=0)+1]; last=[first(2:end)-1;numel(v)];
tiedRanks=repelem((first+last)/2,last-first+1);
ranks=zeros(size(v)); ranks(order)=tiedRanks;
ranks=reshape(ranks,size(x));
end

%% Empirical Quantile
function q=empiricalQuantile(v,p)
v=sort(v(:)); index=1+(numel(v)-1)*p; a=floor(index); b=ceil(index);
q=v(a)+(index-a)*(v(b)-v(a));
end

%% Append Struct
function values=appendStruct(values,row)
if isempty(values)
    values=row;
else
    values(end+1)=orderfields(row,values(1));
end
end

%% Core-Set Selection
function query=codeCoreSet(Xref,Xpool,k,ids,cfg)
n=size(Xpool,1); ids=ids(:);
assert(n>=k && k>=1 && numel(ids)==n && ~isempty(Xref));
assert(size(Xref,2)==size(Xpool,2));
nearestDistance=codeMinimumSquaredDistanceToSet(Xpool,Xref);
nearestDistance(~isfinite(nearestDistance))=-inf;
selected=zeros(k,1); available=true(n,1);
for ell=1:k
    candidateDistance=nearestDistance; candidateDistance(~available)=-inf;
    maximumDistance=max(candidateDistance);
    if ~isfinite(maximumDistance)
        error('Comparison:CodeCoreDistance','No finite candidate distance at step %d.',ell);
    end
    tied=find(candidateDistance>=maximumDistance-cfg.coreTieTol & available);
    chosen=tied(1); selected(ell)=chosen; available(chosen)=false;
    difference=Xpool-Xpool(chosen,:);
    distanceToChosen=sum(difference.^2,2);
    nearestDistance=min(nearestDistance,distanceToChosen);
end
query=ids(selected);
end

%% Minimum Squared Distance
function distance=codeMinimumSquaredDistanceToSet(Xquery,Xreference)
n=size(Xquery,1); distance=inf(n,1);
for j=1:size(Xreference,1)
    difference=Xquery-Xreference(j,:);
    distance=min(distance,sum(difference.^2,2));
end
end
