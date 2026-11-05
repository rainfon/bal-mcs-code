clc;
clearvars;
cfg = struct();
cfg.dataFiles = {'DBS.csv','MPG.csv','CPU.csv','ESL.csv'};
cfg.scriptDir = fileparts(mfilename('fullpath'));
if isempty(cfg.scriptDir), cfg.scriptDir = pwd; end
cfg.dataDir = fullfile(cfg.scriptDir,'Datasets');
addpath(cfg.scriptDir);
cfg.useMexSampler = exist('har_chord_mex','file')==3;
cfg.L = 3;
cfg.Ns = 2000;
cfg.numRepeats = 30;
cfg.numTrajectories = 10;              % R; separate from repeats and Ns
cfg.rounds = 3;
cfg.batchSize = 5;
cfg.refPerClass = 3;
cfg.poolCount = 50;
cfg.xi = 0.4;
cfg.etaSigma = 1;
cfg.msCoverageDistance = 'TP';         % MS has no local distance: fix coverage to TP
cfg.adequacyNormalization = 'recommendation_max'; % revised Eq. (27)
cfg.epsilonAssignment = 1e-6;
cfg.epsilonPhi = 1e-5;
cfg.workers = 16;
cfg.useParallel = true;
cfg.splitSeed = 20261009;
cfg.chainSeed = 271828;
cfg.feedbackSeed = 314159;
cfg.nChains = 4;
cfg.burnIn = 100;
cfg.chainDraws = 200000;               % production draws per chain
cfg.usePreconditioning = true;
cfg.preconditioningMethod = 'hessian';
cfg.directionStrategy = 'coordinate_mixture';
cfg.sphereProbability = 0.2;
cfg.solverTolerance = 1e-9;
cfg.feasibilityTolerance = 1e-9;
cfg.minimumRadius = 1e-12;
cfg.lineIterations = 60;
cfg.barrierMaxIterations = 80;
cfg.barrierNewtonTolerance = 1e-9;
cfg.barrierArmijo = 0.01;
cfg.barrierStepFraction = 0.99;
cfg.queue = []; cfg.verbose = false; cfg.displayDetails = false;
cfg.progressEvery = 1000;
cfg.xiValues = [0.10 0.20 0.30 0.40];
cfg.regionDataset = 1; % DBS; change to 2/3/4 for MPG/CPU/ESL
% A bound on distinct states: initial + each update + R*(k-1) temporary states.
% This is storage capacity only; all statistical settings above are unchanged.
cfg.cacheStates = numel(cfg.xiValues)*(1+cfg.rounds*(1+ ...
    cfg.numTrajectories*(cfg.batchSize-1)));
cfg.verifySampler = true; % short exact comparison, once per paired repeat

assert(all(cfg.xiValues>0 & cfg.xiValues<0.5));
assert(mod(cfg.Ns,cfg.nChains)==0 && mod(cfg.chainDraws,cfg.Ns/cfg.nChains)==0);
assert(exist('linprog','file')==2,'Optimization Toolbox is required.');
oldRng=rng; rngGuard=onCleanup(@()rng(oldRng));
inputs=cell(numel(cfg.dataFiles),1);
for d=1:numel(inputs)
    filename=fullfile(cfg.dataDir,cfg.dataFiles{d});
    assert(exist(filename,'file')==2,'Missing CSV: %s',filename);
    [X,y,preprocessing]=readData(filename);
    [~,dataset]=fileparts(filename);
    savedRng=rng;
    [ref,poolIDs,test]=splitData(y,cfg.refPerClass,cfg.poolCount);
    rng(savedRng);
    fprintf('%s: N=%d, classes=%d, reference=%d, initial pool=%d, fixed test=%d.\n', ...
        dataset,numel(y),max(y),numel(ref),numel(poolIDs),numel(test));
    inputs{d}=struct('X',X,'y',y,'Dataset',dataset,'Preprocessing',preprocessing);
end
if cfg.useParallel
    assert(license('test','Distrib_Computing_Toolbox'),'Parallel Computing Toolbox required.');
    pool=gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers~=cfg.workers, delete(pool); pool=[]; end
    if isempty(pool)
        cluster=parcluster('local'); cluster.NumWorkers=max(cluster.NumWorkers,cfg.workers);
        parpool(cluster,cfg.workers);
    else
        wait(parfevalOnAll(@addpath,0,cfg.scriptDir));
    end
end
config=struct('Model','AVG','Evidence','TP','Batch','TC');
nTasks=numel(inputs)*cfg.numRepeats;
total=nTasks*numel(cfg.xiValues)*cfg.rounds;
queue=parallel.pool.DataQueue;
xiProgress([],total); afterEach(queue,@(message)xiProgress(message,total));
jobs=cell(nTasks,1);
if cfg.useParallel
    parfor task=1:nTasks
        d=ceil(task/cfg.numRepeats); r=mod(task-1,cfg.numRepeats)+1;
        jobs{task}=runXiRepeat(inputs{d},d,r,config,cfg,queue);
    end
else
    for task=1:nTasks
        d=ceil(task/cfg.numRepeats); r=mod(task-1,cfg.numRepeats)+1;
        jobs{task}=runXiRepeat(inputs{d},d,r,config,cfg,queue);
    end
end
for task=1:nTasks
    assert(jobs{task}.Success,'Xi:FailedRepeat','%s repeat %d failed: %s', ...
        jobs{task}.Dataset,jobs{task}.Repeat,jobs{task}.ErrorMessage);
end
AOMAE=zeros(cfg.numRepeats,numel(cfg.xiValues),numel(inputs));
initialCounts=zeros(cfg.numRepeats,numel(cfg.xiValues),3,numel(inputs));
for task=1:nTasks
    job=jobs{task};
    AOMAE(job.Repeat,:,job.DatasetID)=job.AOMAE';
    initialCounts(job.Repeat,:,:,job.DatasetID)=reshape(job.InitialCounts, ...
        [1,numel(cfg.xiValues),3]);
end
mu=reshape(mean(AOMAE,1),numel(cfg.xiValues),numel(inputs))';
sd=reshape(std(AOMAE,0,1),numel(cfg.xiValues),numel(inputs))';
Dataset=repelem(string(cellfun(@(x)x.Dataset,inputs,'UniformOutput',false)),numel(cfg.xiValues));
Xi=repmat(cfg.xiValues(:),numel(inputs),1);
MeanAOMAE=reshape(mu',[],1); SDAOMAE=reshape(sd',[],1);
fprintf('\nAOMAE averages normalized OMAE over the %d post-query rounds; smaller is better.\n',cfg.rounds);
disp(table(Dataset,Xi,MeanAOMAE,SDAOMAE));
% Persist plot-ready values and repeated observations, without data or chains.
plotData=struct();
plotData.FormatVersion=1;
plotData.DatasetNames=cellfun(@(x)x.Dataset,inputs,'UniformOutput',false);
plotData.XiValues=cfg.xiValues;
plotData.RegionDataset=cfg.regionDataset;
plotData.PoolCount=cfg.poolCount;
plotData.AOMAE=AOMAE; % dimensions: repeat x xi x dataset
plotData.InitialCounts=initialCounts; % repeat x xi x action(POS/BND/NEG) x dataset
plotData.MeanAOMAE=mu; % dataset x xi
plotData.SDAOMAE=sd; % dataset x xi; standard deviation across repeated runs
plotData.MeanInitialCounts=reshape(mean(initialCounts,1),numel(cfg.xiValues),3,numel(inputs));
plotData.SDInitialCounts=reshape(std(initialCounts,0,1),numel(cfg.xiValues),3,numel(inputs));
plotData.MACGrid=linspace(0,1,401);
plotData.Alpha=zeros(numel(cfg.xiValues),numel(plotData.MACGrid));
plotData.Beta=plotData.Alpha;
for j=1:numel(cfg.xiValues)
    xi=cfg.xiValues(j); mac=plotData.MACGrid;
    plotData.Alpha(j,:)=(1-xi)*mac./((1-xi)*mac+xi*(1-mac));
    plotData.Beta(j,:)=xi*mac./(xi*mac+(1-xi)*(1-mac));
end
plotData.Summary=table(Dataset,Xi,MeanAOMAE,SDAOMAE);
plotData.Config=cfg;
plotData.Method=config;
plotData.MetricDefinition='Mean over post-query rounds of mean(abs(predicted class-true class))/(q-1).';
fig=plotXiFigure(plotData,cfg.regionDataset);
figFile=fullfile(cfg.scriptDir,[mfilename,'.fig']);
set(findall(fig,'Type','axes'),'Box','on','LineWidth',2);
savefig(fig,figFile);
exportgraphics(fig,fullfile(figureDirs(cfg.scriptDir),'Figure6.pdf'),'ContentType','vector');
fprintf('Figure saved: %s\n',figFile);

function job=runXiRepeat(input,d,r,config,cfg,queue)
job=struct('Success',false,'Dataset',input.Dataset,'DatasetID',d,'Repeat',r,'ErrorMessage','');
try
    rng(keySeed(cfg.splitSeed,d,r,'split'),'twister');
    [ref,pool,test]=splitData(input.y,cfg.refPerClass,cfg.poolCount);
    ref=sort(ref); pool=sort(pool); test=sort(test);
    X=input.X; y=input.y; q=max(y); P=basis(X,cfg.L);
    cfg.progressQueue=queue;
    context=struct('DatasetID',d,'Repeat',r,'X',X,'P',P,'q',q, ...
        'Universe',pool,'nCriteria',size(X,2));
    cache=containers.Map('KeyType','char','ValueType','any'); order={};
    omae=zeros(numel(cfg.xiValues),cfg.rounds);
    counts=zeros(numel(cfg.xiValues),3);
    % Shared exact-state cache and content-indexed seeds preserve paired comparisons.
    for c=1:numel(cfg.xiValues)
        localCfg=cfg; localCfg.xi=cfg.xiValues(c);
        reference=ref; labels=y(ref); active=pool;
        for t=1:cfg.rounds
            [base,order,~]=getState(reference,labels,context,localCfg,cache,order,true);
            state=realizeState(base,config,active,context,localCfg);
            if t==1
                for a=1:3, counts(c,a)=sum(state.Action==a); end
            end
            [query,order,~,~]=selectBatch(state,reference,labels,active,config, ...
                context,localCfg,cache,order,t);
            assert(numel(query)==min(cfg.batchSize,numel(active)) && numel(unique(query))==numel(query));
            assert(all(ismember(query,active)) && ~any(ismember(query,test)));
            reference=[reference;query]; labels=[labels;y(query)];
            [reference,ix]=sort(reference); labels=labels(ix);
            active=setdiff(active,query,'stable');
            [updated,order,~]=getState(reference,labels,context,localCfg,cache,order,true);
            model=representativeModel(updated,config.Model,active,context,localCfg);
            pred=assignClasses(P(test,:)*model(1:updated.Region.p),model(updated.Region.p+1:end),q);
            omae(c,t)=mean(abs(pred-y(test)))/(q-1);
            send(queue,struct('Kind','round','Count',1,'Dataset',input.Dataset, ...
                'Repeat',r,'Config',c,'Round',t));
        end
    end
    job.Success=true; job.AOMAE=mean(omae,2); job.OMAE=omae; job.InitialCounts=counts;
catch ME
    job.ErrorMessage=sprintf('%s: %s',ME.identifier,ME.message);
    send(queue,struct('Kind','failure','Dataset',input.Dataset,'Repeat',r,'Error',job.ErrorMessage));
end
end

function xiProgress(message,total)
persistent completed timerStart lastPercent
if isempty(message)
    completed=0; timerStart=tic; lastPercent=-1;
    chainProgress('reset');
    fprintf('Progress: 0%% | Elapsed: 00:00:00 | Remaining: --:--:--\n');
    return;
end
if strcmp(message.Kind,'failure')
    fprintf(2,'%s repeat %d: %s\n',message.Dataset,message.Repeat,message.Error); return;
end
chain=chainProgress('update',message);
isRound=strcmp(message.Kind,'round');
if isRound, completed=completed+message.Count; end
if ~isRound && ~chain.changed, return; end
percent=min(100,floor(100*completed/max(total,1)));
if percent==lastPercent && ~chain.changed, return; end
elapsed=toc(timerStart);
if completed>0
    remaining=formatDuration(elapsed*(total-completed)/completed);
else
    remaining='--:--:--';
end
if chain.percent<0
    fprintf('Progress: %d%% | Elapsed: %s | Remaining: %s\n', ...
        percent,formatDuration(elapsed),remaining);
else
    fprintf('Progress: %d%% | Chain: %d%% | Elapsed: %s | Remaining: %s\n', ...
        percent,chain.percent,formatDuration(elapsed),remaining);
end
lastPercent=percent;
end

function text=formatDuration(seconds)
seconds=max(0,round(seconds));
text=sprintf('%02d:%02d:%02d',floor(seconds/3600),floor(mod(seconds,3600)/60),mod(seconds,60));
end

function [base,order,newFit]=getState(ref,labels,context,cfg,cache,order,isRealState)
[ref,ix]=sort(ref); labels=labels(ix);
assert(numel(unique(ref))==numel(ref),'Duplicate reference alternative.');
key=sprintf('%d:%d;',reshape([ref(:),labels(:)]',[],1));
if isKey(cache,key)
    base=cache(key); newFit=false;
    if isRealState && ~base.IsRealState, base.IsRealState=true; cache(key)=base; end
    position=find(strcmp(order,key),1); order(position)=[]; order{end+1}=key;
    return;
end
newFit=true;
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
    rng(seeds(ch),'twister');
    if cfg.verifySampler && cache.Count==0 && ch==1
        verifySamplerExact(R,starts(:,ch),samplingCfg);
    end
    [chain,audit]=sampleChain(R,starts(:,ch),samplingCfg);
    assert(audit.NumericalFailures==0,'Xi:NumericalFailure', ...
        'Sampler numerical failures; inspect geometry. No chain shortening or resampling.');
    M(:,(ch-1)*perChain+(1:perChain))=chain';
    audits{ch}=audit;
end
ids=[context.Universe;ref]; allP=context.P(ids,:); U=allP*M(1:R.p,:);
assignment=ones(size(U));
for h=1:R.q-1, assignment=assignment+(U>M(R.p+h,:)); end
support=zeros(numel(ids),R.q);
for h=1:R.q, support(:,h)=mean(assignment==h,2); end
% Deviations are computed from exactly Eq. (25).
robust=zeros(numel(ids),R.q);
for h=1:R.q
    rho=classDeviation(U,M,R,h); robust(:,h)=max(rho,[],2);
end
rad=min(robust,[],2);
Dtp=thresholdDistance(U,U,M,R,cfg);
nPool=numel(context.Universe); localY=labels;
activeRows=find(~ismember(context.Universe,ref));
ptp=integratedEvidence(support,Dtp(:,nPool+1:end),localY,cfg,activeRows);
% Cache only quantities actually consumed by fixed AVG-TP-TC. Keeping the
% original products/reductions above preserves their floating-point order.
base=struct('Region',struct('p',R.p,'q',R.q),'AverageModel',mean(M,2), ...
    'IDs',ids,'TP',ptp,'Dtp',Dtp,'RAD',rad,'Seeds',seeds, ...
    'StateKey',key,'IsRealState',isRealState);
assert(cache.Count<cfg.cacheStates,'Xi:CacheBound', ...
    'Distinct-state bound exceeded; inspect the experimental loop.');
cache(key)=base; order{end+1}=key;
end
function state=realizeState(base,config,active,context,cfg)
[found,rows]=ismember(active,base.IDs); assert(all(found));
model=representativeModel(base,config.Model,active,context,cfg);
U=context.P(active,:)*model(1:base.Region.p);
hat=assignClasses(U,model(base.Region.p+1:end),context.q);
evidence=base.(config.Evidence); evidence=evidence(rows,:);
score=adequacyScore(evidence,hat,cfg);
b=[0;model(base.Region.p+1:end);1];
lo=b(hat); hi=b(hat+1); width=hi-lo;
mac=ones(size(U)); valid=width>0;
mac(valid)=1-2*min(U(valid)-lo(valid),hi(valid)-U(valid))./width(valid);
mac=max(0,min(1,mac));
% Risk implementation agrees with Eq. (32) internally and defines endpoints:
% exact ties are resolved retention -> deferment -> query (POS priority).
risks=[mac.*(1-score),cfg.xi*((1-mac).*score+mac.*(1-score)),(1-mac).*score];
[~,action]=min(risks,[],2);
state=struct('IDs',active,'Rows',rows,'Model',model,'Evidence',evidence, ...
    'Score',score,'MAC',mac,'Action',action,'RAD',base.RAD(rows),'Base',base);
end
function model=representativeModel(base,variant,active,context,cfg) %#ok<INUSD>
assert(strcmp(variant,'AVG'),'This accelerated runner is specialized for AVG-TP-TC.');
model=base.AverageModel;
end
function score=adequacyScore(p,hat,cfg)
q=size(p,2); distance=abs(hat(:)-(1:q));
if strcmp(cfg.adequacyNormalization,'recommendation_max')
    denominator=max(hat(:)-1,q-hat(:));
else, error('Use the revised recommendation_max normalization.'); end
score=1-sum(p.*distance,2)./denominator;
assert(all(score>=-1e-12&score<=1+1e-12)); score=max(0,min(1,score));
end
function labels=assignClasses(U,b,q)
labels=ones(size(U));
for h=1:q-1, labels=labels+(U>b(h)); end
end
function rho=classDeviation(U,M,R,h)
if h==1, lower=zeros(size(U)); else, lower=M(R.p+h-1,:)+R.epsilon-U; end
if h==R.q, upper=zeros(size(U)); else, upper=U-M(R.p+h,:); end
rho=max(0,max(lower,upper));
end
function D=thresholdDistance(Ua,Ub,M,R,cfg) %#ok<INUSD>
% Exact Eq. (26). For identical row sets, evaluate only the strict triangle.
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
function p=integratedEvidence(ms,D,labels,cfg,activeRows)
% Bandwidth uses CURRENT pool-to-reference distances, not test features/labels.
sigma=distanceBandwidth(D(activeRows,:),cfg);
K=exp(-(D/sigma).^2); W=sum(K,2); local=zeros(size(ms));
for h=1:size(ms,2), local(:,h)=sum(K(:,labels==h),2); end
p=(ms+local)./(1+W); % exactly Eqs. (22)-(23), including W=0
assert(max(abs(sum(p,2)-1))<1e-10);
end
function sigma=distanceBandwidth(D,cfg)
positive=D(D>0);
if isempty(positive), sigma=eps; else, sigma=cfg.etaSigma*median(positive); end
end
function distance=coverageDistance(base,config,cfg) %#ok<INUSD>
assert(strcmp(config.Evidence,'TP'),'This accelerated runner uses TP evidence.');
distance=base.Dtp;
end
function [query,order,nFits,nHits]=selectBatch(state,ref,labels,active,config,context,cfg,cache,order,roundID)
k=min(cfg.batchSize,numel(active)); query=[]; nFits=0; nHits=0;
% Eq. (35)/(39) applies when NEG alone can fill the batch. Otherwise include
% ALL NEG, then fill from BND and POS by RAD with ID as deterministic tie break.
negative=find(state.Action==3);
if numel(negative)<k || strcmp(config.Batch,'TR')
    for action=[3,2,1]
        rows=find(state.Action==action);
        ranking=sortrows([-state.RAD(rows),state.IDs(rows)],[1,2]);
        take=min(k-numel(query),size(ranking,1));
        query=[query;ranking(1:take,2)]; %#ok<AGROW>
        if numel(query)==k, return; end
    end
elseif strcmp(config.Batch,'SC')
    D=coverageDistance(state.Base,config,cfg); ix=state.Rows(negative);
    demandIDs=state.IDs(negative); subD=D(ix,ix);
    [~,currentRows]=ismember(active,state.Base.IDs);
    nPool=numel(context.Universe);
    sigma=distanceBandwidth(D(currentRows,nPool+1:end),cfg);
    K=exp(-(subD/sigma).^2);
    weights=normalizedWeights(state.RAD(negative));
    query=greedyCoverage(K,weights,k,demandIDs);
elseif strcmp(config.Batch,'TC')
    eligible=sort(state.IDs(negative));
    demandIDs=zeros(cfg.numTrajectories,k); importance=zeros(cfg.numTrajectories,k);
    % Kernel columns are computed in their own temporary state. Every eligible
    % candidate is evaluated now; restrict rows to the trajectory union later.
    kernel=zeros(numel(eligible),cfg.numTrajectories*k);
    uniformStream=RandStream('mt19937ar','Seed',keySeed(cfg.feedbackSeed, ...
        context.DatasetID,context.Repeat,sprintf('round=%d',roundID)));
    uniformInputs=rand(uniformStream,cfg.numTrajectories,k);
    for trajectory=1:cfg.numTrajectories
        tmpRef=ref; tmpLabels=labels; visited=[];
        for position=1:k
            [base,order,newFit]=getState(tmpRef,tmpLabels,context,cfg,cache,order,false);
            nFits=nFits+newFit; nHits=nHits+~newFit;
            remaining=setdiff(eligible,visited,'stable');
            [found,rows]=ismember(remaining,base.IDs); assert(all(found));
            ranked=sortrows([-base.RAD(rows),remaining],[1,2]); chosen=ranked(1,2);
            [~,row]=ismember(chosen,base.IDs);
            demandIDs(trajectory,position)=chosen; importance(trajectory,position)=base.RAD(row);
            [found,allRows]=ismember(eligible,base.IDs); assert(all(found));
            distance=coverageDistance(base,config,cfg);
            % Temporary bandwidth follows Section 3.3: remaining unqueried pool
            % to temporary references, same kernel for evidence and coverage.
            temporaryActive=setdiff(context.Universe,tmpRef,'stable');
            [~,activeRows]=ismember(temporaryActive,base.IDs);
            nPool=numel(context.Universe);
            sigma=distanceBandwidth(distance(activeRows,nPool+1:end),cfg);
            column=(trajectory-1)*k+position;
            kernel(:,column)=exp(-(distance(allRows,row)/sigma).^2);
            p=base.(config.Evidence); cumulative=cumsum(p(row,:)); cumulative(end)=1;
            hypothetical=find(uniformInputs(trajectory,position)<cumulative,1);
            tmpRef=[tmpRef;chosen]; tmpLabels=[tmpLabels;hypothetical]; visited=[visited;chosen];
        end
    end
    universe=unique(demandIDs(:)); [~,universeRows]=ismember(universe,eligible);
    weights=zeros(cfg.numTrajectories*k,1);
    for trajectory=1:cfg.numTrajectories
        weights((trajectory-1)*k+(1:k))=normalizedWeights(importance(trajectory,:)');
    end
    query=greedyCoverage(kernel(universeRows,:),weights,k,universe);
else, error('Unknown batch variant: %s',config.Batch);
end
end
function w=normalizedWeights(v)
if sum(v)>0, w=v/sum(v); else, w=ones(size(v))/numel(v); end
end
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
function seed=keySeed(base,d,r,key)
% Stable content-indexed random streams: identical information states share
% samples. No dependence on configuration, worker, order, cache hit, or round.
% Separate base seeds distinguish splitting, chains, and hypothetical labels.
prime=2147483647;
seed=mod(double(base)+104729*d+13007*r,prime);
bytes=double(char(key));
for j=1:numel(bytes), seed=mod(seed*131+bytes(j),prime); end
seed=max(1,floor(seed));
end
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
function P=basis(X,L)
P=zeros(size(X,1),size(X,2)*L);
for j=1:size(X,2)
    for k=1:L, P(:,(j-1)*L+k)=max(0,min(1,L*X(:,j)-(k-1))); end
end
end
function [R,info]=makeRegion(X,y,q,cfg)
P=basis(X,cfg.L); p=size(P,2); n=numel(y); nv=p+q-1;
eq=[ones(1,p),zeros(1,q-1)]; B=null(eq); % orthonormal tangent basis
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
function [x,fval,report]=solveLP(f,A,b,Aeq,beq,lb,ub,cfg)
persistent opts cachedTolerance
if isempty(opts) || ~isequal(cachedTolerance,cfg.solverTolerance)
    opts=optimoptions('linprog','Display','none', ...
        'ConstraintTolerance',cfg.solverTolerance,'OptimalityTolerance',cfg.solverTolerance);
    cachedTolerance=cfg.solverTolerance;
end
[x,fval,flag,output]=linprog(f,sparse(A),b,sparse(Aeq),beq,lb,ub,opts);
assert(flag>0&&~isempty(x)&&all(isfinite(x)),'HAR:LP','linprog failed: %s',output.message);
res=max([0;A*x-b;abs(Aeq*x-beq);lb-x]);
if ~isempty(ub), res=max([res; x-ub]); end
assert(res<=cfg.feasibilityTolerance,'HAR:LPResidual','LP residual %g exceeds tolerance.',res);
report=struct('ExitFlag',flag,'Message',output.message,'MaxPrimalResidual',res);
end
%% 6. Sampling and geometric preconditioning
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
function [R,report]=preparePreconditioner(R,starts,seeds,cfg) %#ok<INUSD>
% Fixed Hessian geometry; no covariance pilots or alternative geometry paths.
assert(strcmp(cfg.preconditioningMethod,'hessian'),'Only fixed Hessian geometry is supported.');
if cfg.usePreconditioning
    [R,report]=prepareHessianGeometry(R,cfg);
else
    R.DirectionBasis=R.B; report=table();
end
end
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
    if ~accepted, break; end % retain verified interior point, report nonconvergence
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
if ~converged
    fprintf('Geometry center solve stopped early; retaining the verified interior point.\n');
end
end
function [chain,audit]=sampleChain(R,start,cfg)
t0=tic; x=start;
keepThin=cfg.chainDraws/(cfg.Ns/cfg.nChains);
chain=zeros(cfg.Ns/cfg.nChains,R.nVar); kept=0;
minWidth=inf; maxWidth=0;
failures=0; minStep=inf; maxResidual=0; maxEq=0; maxLoss=0;
[sr,er,lr]=residuals(x,R);
sinceReport=0;
if isfield(R,'DirectionBasis'), directionBasis=R.DirectionBasis; else, directionBasis=R.B; end
useCoordinates=strcmp(cfg.directionStrategy,'coordinate_mixture');
coordinateDirections=directionBasis;
for j=1:size(directionBasis,2)
    coordinateDirections(:,j)=directionBasis(:,j)/norm(directionBasis(:,j));
end
% Compute each product with the SAME vector shape and multiplication used
% inside chordReference; values do not depend on the current chain position.
useMex=isfield(cfg,'useMexSampler') && cfg.useMexSampler;
if useMex && exist('har_chord_mex','file')~=3
    error('Xi:MissingMex','cfg.useMexSampler is true but har_chord_mex is not compiled for this platform.');
end
S=R.S; sb=R.sb; Phi=R.Phi; yy=R.y; p=R.p; qq=R.q;
epsilon=R.epsilon; cap=R.cap; feasTol=cfg.feasibilityTolerance; lineIterations=cfg.lineIterations;
if ~useMex
    coordinateDen=zeros(size(R.S,1),size(coordinateDirections,2));
    coordinateUd=zeros(size(R.Phi,1),size(coordinateDirections,2));
    for j=1:size(coordinateDirections,2)
        d=coordinateDirections(:,j);
        coordinateDen(:,j)=R.S*d;
        coordinateUd(:,j)=R.Phi*d(1:R.p);
    end
end
hasQueue=isfield(cfg,'queue') && ~isempty(cfg.queue);
hasProgressContext=isfield(cfg,'progressContext');
reportClock=tic;
for t=1:(cfg.burnIn+cfg.chainDraws)
    if useCoordinates && ...
            (cfg.sphereProbability==0 || rand>cfg.sphereProbability)
        j=randi(size(directionBasis,2)); d=coordinateDirections(:,j);
        if ~useMex, den=coordinateDen(:,j); Ud=coordinateUd(:,j); end
    else
        v=randn(size(directionBasis,2),1); d=directionBasis*v; d=d/norm(d);
        if ~useMex, den=R.S*d; Ud=R.Phi*d(1:R.p); end
    end
    if useMex
        [left,right]=har_chord_mex(x,d,S,sb,Phi,yy,p,qq,epsilon,cap,feasTol,lineIterations);
    else
        [left,right]=chord(x,d,R,cfg,den,Ud);
    end
    width=right-left;
    assert(isfinite(width)&&width>=0,'HAR:Chord','Invalid chord.');
    minWidth=min(minWidth,width); maxWidth=max(maxWidth,width);
    if width<=64*eps(max(1,norm(x)))
        failures=failures+1; % keep x; recorded numerical self-loop, no redraw
    else
        u=rand; candidate=x+(left+u*width)*d;
        [srCandidate,eqCandidate,lrCandidate]=residuals(candidate,R);
        if srCandidate>cfg.feasibilityTolerance || ...
                eqCandidate>cfg.feasibilityTolerance || lrCandidate>0 || ~all(isfinite(candidate))
            failures=failures+1;
        else
            minStep=min(minStep,norm(candidate-x)); x=candidate;
            sr=srCandidate; er=eqCandidate; lr=lrCandidate;
        end
    end
    maxResidual=max(maxResidual,sr); maxEq=max(maxEq,er); maxLoss=max(maxLoss,lr);
    if t>cfg.burnIn && mod(t-cfg.burnIn,keepThin)==0
        kept=kept+1; chain(kept,:)=x';
    end
    if hasQueue
       sinceReport=sinceReport+1;
       if (mod(t,cfg.progressEvery)==0 && toc(reportClock)>=15) || t==cfg.burnIn+cfg.chainDraws
           if hasProgressContext
               packet=cfg.progressContext; packet.Count=sinceReport; packet.Transition=t; packet.TotalTransitions=cfg.burnIn+cfg.chainDraws; send(cfg.queue,packet);
           else
               send(cfg.queue,sinceReport);
           end
           sinceReport=0; reportClock=tic;
       end
   end
   if cfg.verbose && cfg.chainDraws>=10000 && mod(t,10000)==0
        fprintf('  transitions %d/%d\n',t,cfg.burnIn+cfg.chainDraws);
    end
end
audit=struct('Seconds',toc(t0),'Transitions',cfg.burnIn+cfg.chainDraws,'BurnIn',cfg.burnIn, ...
    'ContinuousDraws',cfg.chainDraws,'StoredDraws',kept,'StorageThin',keepThin,'NumericalFailures',failures,'MinChord',minWidth, ...
    'MaxChord',maxWidth,'MinStep',minStep,'MaxStructuralResidual',maxResidual, ...
    'MaxEqualityResidual',maxEq,'MaxLossResidual',maxLoss);
end
function [left,right]=chord(x,d,R,cfg,den,Ud)
slack=R.sb-R.S*x;
assert(all(slack>=-cfg.feasibilityTolerance),'HAR:Outside','Start outside structure.');
slack=max(slack,0); left=-inf; right=inf;
positive=den>0; negative=den<0;
if any(positive), right=min(slack(positive)./den(positive)); end
if any(negative), left=max(slack(negative)./den(negative)); end
assert(isfinite(left)&&isfinite(right)&&left<=0&&right>=0,'HAR:Chord','Unbounded/invalid structural chord.');
if isempty(R.y), return; end
% Cache affine loss branches once per direction. This is the same convex
% piecewise-linear loss, not a surrogate or a relaxation of the region.
U=R.Phi*x(1:R.p);
b=x(R.p+1:end); bd=d(R.p+1:end);
n=numel(R.y); lo=zeros(n,1); up=lo; lod=lo; upd=lo;
ids=R.y>1;
lo(ids)=b(R.y(ids)-1)+R.epsilon-U(ids);
lod(ids)=bd(R.y(ids)-1)-Ud(ids);
ids=R.y<R.q;
up(ids)=U(ids)-b(R.y(ids));
upd(ids)=Ud(ids)-bd(R.y(ids));
originalLeft=left; originalRight=right;
if lineLoss(left,lo,up,lod,upd)>R.cap
    left=affineEndpoint(left,lo,up,lod,upd,R.cap,cfg.lineIterations);
end
if lineLoss(right,lo,up,lod,upd)>R.cap
    right=affineEndpoint(right,lo,up,lod,upd,R.cap,cfg.lineIterations);
end
% Recheck with the original expression; fall back on rare rounding conflicts.
if lossValue(x+left*d,R)>R.cap
    left=bisectEndpoint(originalLeft,x,d,R,cfg);
end
if lossValue(x+right*d,R)>R.cap
    right=bisectEndpoint(originalRight,x,d,R,cfg);
end
end
function value=lineLoss(t,lo,up,lod,upd)
value=sum(max(0,max(lo+t*lod,up+t*upd)));
end
function endpoint=affineEndpoint(outside,lo,up,lod,upd,cap,iterations)
% The line loss is a convex sum of maxima of THREE affine branches.
% At an outside point, the active affine tangent crosses the cap on or beyond
% the true boundary. Iterating moves inward through affine pieces. Return only
% a feasible endpoint; fall back to the original bisection if rounding stalls.
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
function endpoint=bisectEndpoint(outside,x,d,R,cfg)
inside=0;
assert(lossValue(x,R)<=R.cap,'HAR:LossStart','Current point exceeds exact loss cap.');
for k=1:cfg.lineIterations
    middle=0.5*(outside+inside);
    if middle==outside||middle==inside, break; end
    if lossValue(x+middle*d,R)<=R.cap, inside=middle; else, outside=middle; end
end
endpoint=inside; % conservative numerical endpoint, no deliberate guard
end
function value=lossValue(x,R)
if isempty(R.y), value=0; return; end
U=R.Phi*x(1:R.p); b=x(R.p+1:end); lower=zeros(numel(R.y),1); upper=lower;
ids=R.y>1; lower(ids)=b(R.y(ids)-1)+R.epsilon-U(ids);
ids=R.y<R.q; upper(ids)=U(ids)-b(R.y(ids));
value=sum(max(0,max(lower,upper)));
end
function [s,e,l]=residuals(x,R)
s=max([0;R.S*x-R.sb]); e=max([0;abs(R.eq*x-R.eqValue)]); l=max(0,lossValue(x,R)-R.cap);
end
function assertFeasible(x,R,cfg)
[s,e,l]=residuals(x,R);
assert(all(isfinite(x))&&max([s,e,l])<=cfg.feasibilityTolerance, ...
    'HAR:Feasibility','Invalid parameter: structural=%g equality=%g loss=%g',s,e,l);
end

function values=appendStruct(values,row)
if isempty(values)
    values=row;
else
    values(end+1)=orderfields(row,values(1));
end
end

function verifySamplerExact(R,start,cfg)
% Check identical retained values and RNG state on a short chain prefix.
% This is not a sampling/convergence diagnostic and does not tune parameters.
% The production chain still uses all original burn-in/draws/thinning values.
entryRng=rng; restore=onCleanup(@()rng(entryRng)); %#ok<NASGU>
check=cfg; check.chainDraws=128; check.Ns=128*cfg.nChains;
check.queue=[]; check.verbose=false;
[fast,fastAudit]=sampleChain(R,start,check); fastRng=rng;
rng(entryRng);
[reference,referenceAudit]=sampleChainReference(R,start,check); referenceRng=rng;
assert(isequal(fast,reference) && isequal(fastRng,referenceRng) && ...
    fastAudit.NumericalFailures==referenceAudit.NumericalFailures, ...
    'Xi:ExactSamplerMismatch', ...
    'Optimized/reference sampler mismatch; production run stopped.');
end

function [chain,audit]=sampleChainReference(R,start,cfg)
t0=tic; x=start;
keepThin=cfg.chainDraws/(cfg.Ns/cfg.nChains);
chain=zeros(cfg.Ns/cfg.nChains,R.nVar); kept=0;
minWidth=inf; maxWidth=0;
failures=0; minStep=inf; maxResidual=0; maxEq=0; maxLoss=0;
[sr,er,lr]=residuals(x,R);
sinceReport=0;
if isfield(R,'DirectionBasis'), directionBasis=R.DirectionBasis; else, directionBasis=R.B; end
useCoordinates=strcmp(cfg.directionStrategy,'coordinate_mixture');
coordinateDirections=directionBasis;
for j=1:size(directionBasis,2)
    coordinateDirections(:,j)=directionBasis(:,j)/norm(directionBasis(:,j));
end
reportClock=tic;
for t=1:(cfg.burnIn+cfg.chainDraws)
    if useCoordinates && ...
            (cfg.sphereProbability==0 || rand>cfg.sphereProbability)
        j=randi(size(directionBasis,2)); d=coordinateDirections(:,j);
    else
        v=randn(size(directionBasis,2),1); d=directionBasis*v; d=d/norm(d);
    end
    [left,right]=chordReference(x,d,R,cfg); width=right-left;
    assert(isfinite(width)&&width>=0,'HAR:Chord','Invalid chord.');
    minWidth=min(minWidth,width); maxWidth=max(maxWidth,width);
    if width<=64*eps(max(1,norm(x)))
        failures=failures+1; % keep x; recorded numerical self-loop, no redraw
    else
        u=rand; candidate=x+(left+u*width)*d;
        [srCandidate,eqCandidate,lrCandidate]=residuals(candidate,R);
        if srCandidate>cfg.feasibilityTolerance || ...
                eqCandidate>cfg.feasibilityTolerance || lrCandidate>0 || ~all(isfinite(candidate))
            failures=failures+1;
        else
            minStep=min(minStep,norm(candidate-x)); x=candidate;
            sr=srCandidate; er=eqCandidate; lr=lrCandidate;
        end
    end
    maxResidual=max(maxResidual,sr); maxEq=max(maxEq,er); maxLoss=max(maxLoss,lr);
    if t>cfg.burnIn && mod(t-cfg.burnIn,keepThin)==0
        kept=kept+1; chain(kept,:)=x';
    end
    if isfield(cfg,'queue') && ~isempty(cfg.queue)
       sinceReport=sinceReport+1;
       if (mod(t,cfg.progressEvery)==0 && toc(reportClock)>=15) || t==cfg.burnIn+cfg.chainDraws
           if isfield(cfg,'progressContext')
               packet=cfg.progressContext; packet.Count=sinceReport; packet.Transition=t; packet.TotalTransitions=cfg.burnIn+cfg.chainDraws; send(cfg.queue,packet);
           else
               send(cfg.queue,sinceReport);
           end
           sinceReport=0; reportClock=tic;
       end
   end
   if cfg.verbose && cfg.chainDraws>=10000 && mod(t,10000)==0
        fprintf('  transitions %d/%d\n',t,cfg.burnIn+cfg.chainDraws);
    end
end
audit=struct('Seconds',toc(t0),'Transitions',cfg.burnIn+cfg.chainDraws,'BurnIn',cfg.burnIn, ...
    'ContinuousDraws',cfg.chainDraws,'StoredDraws',kept,'StorageThin',keepThin,'NumericalFailures',failures,'MinChord',minWidth, ...
    'MaxChord',maxWidth,'MinStep',minStep,'MaxStructuralResidual',maxResidual, ...
    'MaxEqualityResidual',maxEq,'MaxLossResidual',maxLoss);
end
function [left,right]=chordReference(x,d,R,cfg)
slack=R.sb-R.S*x;
assert(all(slack>=-cfg.feasibilityTolerance),'HAR:Outside','Start outside structure.');
slack=max(slack,0); den=R.S*d; left=-inf; right=inf;
positive=den>0; negative=den<0;
if any(positive), right=min(slack(positive)./den(positive)); end
if any(negative), left=max(slack(negative)./den(negative)); end
assert(isfinite(left)&&isfinite(right)&&left<=0&&right>=0,'HAR:Chord','Unbounded/invalid structural chord.');
if isempty(R.y), return; end
% Cache affine loss branches once per direction. This is the same convex
% piecewise-linear loss, not a surrogate or a relaxation of the region.
U=R.Phi*x(1:R.p); Ud=R.Phi*d(1:R.p);
b=x(R.p+1:end); bd=d(R.p+1:end);
n=numel(R.y); lo=zeros(n,1); up=lo; lod=lo; upd=lo;
ids=R.y>1;
lo(ids)=b(R.y(ids)-1)+R.epsilon-U(ids);
lod(ids)=bd(R.y(ids)-1)-Ud(ids);
ids=R.y<R.q;
up(ids)=U(ids)-b(R.y(ids));
upd(ids)=Ud(ids)-bd(R.y(ids));
originalLeft=left; originalRight=right;
if lineLoss(left,lo,up,lod,upd)>R.cap
    left=affineEndpoint(left,lo,up,lod,upd,R.cap,cfg.lineIterations);
end
if lineLoss(right,lo,up,lod,upd)>R.cap
    right=affineEndpoint(right,lo,up,lod,upd,R.cap,cfg.lineIterations);
end
% Recheck with the original expression; fall back on rare rounding conflicts.
if lossValue(x+left*d,R)>R.cap
    left=bisectEndpoint(originalLeft,x,d,R,cfg);
end
if lossValue(x+right*d,R)>R.cap
    right=bisectEndpoint(originalRight,x,d,R,cfg);
end
end

function fig=plotXiFigure(plotData,regionDataset)
mu=plotData.MeanAOMAE; sd=plotData.SDAOMAE;
xi=plotData.XiValues(:)'; names=cellstr(string(plotData.DatasetNames));
colors=[169 215 240;48 170 208;253 192 27;255 117 4]/255;
regionColors=colors([1 3 4],:);
font='Arial'; fs=30; lw=2;
assert(numel(xi)==4,'This four-color layout expects four xi values.');
assert(all(isfinite(mu(:))) & all(isfinite(sd(:))) & all(sd(:)>=0));
fig=figure('Visible','on','Color','w','Name','Three-way policy sensitivity','NumberTitle','off', ...
    'Units','pixels','Position',[40 100 2100 650]);
positions=[0.065 0.20 0.255 0.68;0.395 0.20 0.255 0.68;0.725 0.20 0.255 0.68];
axesList=gobjects(1,3);
for j=1:3
    axesList(j)=axes('Parent',fig,'Units','normalized','Position',positions(j,:), ...
        'FontName',font,'FontSize',fs,'LineWidth',lw,'Box','on', ...
        'TickDir','out','TickLength',[0.010 0.010],'XGrid','off','YGrid','off', ...
        'Layer','top','PositionConstraint','innerposition');
    hold(axesList(j),'on');
end
labels=arrayfun(@(x)sprintf('$\\xi = %.1f$',x),xi,'UniformOutput',false);

ax=axesList(1); curves=gobjects(4,1); mac=plotData.MACGrid;
for j=1:4
    curves(j)=plot(ax,mac,plotData.Alpha(j,:),'-','Color',colors(j,:),'LineWidth',lw);
    plot(ax,mac,plotData.Beta(j,:),'--','Color',colors(j,:),'LineWidth',lw, ...
        'HandleVisibility','off');
end
xlabel(ax,'MAC_i','Interpreter','tex'); ylabel(ax,'Threshold');
xlim(ax,[0 1]); ylim(ax,[0 1]); xticks(ax,0:0.2:1); yticks(ax,0:0.2:1);
text(ax,0.98,0.045,'Solid: \alpha_i   Dashed: \beta_i','Units','normalized', ...
    'HorizontalAlignment','right','FontName',font,'FontSize',fs,'Interpreter','tex');
placeTitle(ax,'(a) Three-way thresholds');
insideLegend(ax,curves,labels,'northwest',1,'vertical',font,fs);

ax=axesList(2); d=regionDataset;
regionMu=plotData.MeanInitialCounts(:,:,d);
regionSD=plotData.SDInitialCounts(:,:,d);
assert(isequal(size(regionMu),[4 3]) && isequal(size(regionSD),[4 3]));
assert(all(isfinite(regionMu(:))) & all(isfinite(regionSD(:))) & all(regionSD(:)>=0));
bars=spacedBars(ax,regionMu,regionSD,regionColors,0.23,0.185,lw);
xticks(ax,1:4); xticklabels(ax,compose('%.1f',xi)); xlim(ax,[0.48 4.52]);
xlabel(ax,'$\xi$','Interpreter','latex'); ylabel(ax,'Number of alternatives');
upper=max(regionMu(:)+regionSD(:)); lower=min(regionMu(:)-regionSD(:));
top=max(5,5*ceil(1.22*upper/5)); bottom=min(0,5*floor(lower/5));
ylim(ax,[bottom top]); yticks(ax,bottom:5:top);
placeTitle(ax,sprintf('(b) Initial partition: %s',names{d}));
insideLegend(ax,bars,{'POS','BND','NEG'},'northeast',3,'horizontal',font,fs);

ax=axesList(3);
bars=spacedBars(ax,mu,sd,colors,0.18,0.145,lw);
n=numel(names); xticks(ax,1:n); xticklabels(ax,names); xlim(ax,[0.48 n+0.52]);
xlabel(ax,'Dataset'); ylabel(ax,'AOMAE');
upper=max(mu(:)+sd(:)); lower=min(mu(:)-sd(:));
step=niceStep(max(0.01,1.08*upper)/5);
top=max(step,step*ceil(1.08*upper/step)); bottom=min(0,step*floor(lower/step));
ylim(ax,[bottom top]); yticks(ax,bottom:step:top); ytickformat(ax,'%.2f');
placeTitle(ax,'(c) AOMAE');
insideLegend(ax,bars,labels,'northeast',2,'horizontal',font,fs);

set(findall(fig,'-property','FontName'),'FontName',font);
set(findall(fig,'-property','FontSize'),'FontSize',fs);
set(findall(fig,'Type','text'),'FontWeight','normal');
drawnow;
end

function bars=spacedBars(ax,mu,sd,colors,spacing,width,lw)
[n,m]=size(mu); offsets=((1:m)-(m+1)/2)*spacing;
bars=gobjects(m,1);
for j=1:m
    x=(1:n)'+offsets(j);
    bars(j)=bar(ax,x,mu(:,j),width,'FaceColor',colors(j,:), ...
        'EdgeColor',colors(j,:),'LineWidth',lw);
    errorbar(ax,x,mu(:,j),sd(:,j),'Color',0.48*colors(j,:), ...
        'LineStyle','none','LineWidth',lw,'CapSize',9,'HandleVisibility','off');
end
end

function placeTitle(ax,label)
t=title(ax,label,'FontWeight','normal','Interpreter','none');
t.Units='normalized'; t.Position=[0.5 1.035 0];
end

function insideLegend(ax,handles,labels,location,columns,orientation,font,fs)
if any(contains(labels,'$')), interpreter='latex'; else, interpreter='tex'; end
lg=legend(ax,handles,labels,'Location',location,'Orientation',orientation, ...
    'NumColumns',columns,'Box','off','Interpreter',interpreter, ...
    'FontName',font,'FontSize',fs,'Color','none');
lg.ItemTokenSize=[22 14];
end

function step=niceStep(raw)
scale=10^floor(log10(max(raw,realmin)));
candidates=[1 2 2.5 5 10]*scale;
step=candidates(find(candidates>=raw,1));
end
