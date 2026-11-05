function RQ3_Common_Test

%% Settings
cfg.scriptDir=fileparts(mfilename('fullpath'));
if isempty(cfg.scriptDir), cfg.scriptDir=pwd; end
cfg.outputDir=cfg.scriptDir;
cfg.texName=[mfilename,'.tex'];
cfg.figureName=mfilename;
cfg.protocol='CEV_LEV_MMG_AVG_TP_TC_ref3579_budget0to9_v2';
cfg.workers=16;
cfg.adequacyMode='ordinal';
cfg.adequacyNormalization='recommendation_max';
cfg.variantLabel='AVG-TP-TC';
cfg.priorityOrder=[3 2 1];
cfg.poolCount=50;
cfg.batchSize=3;
cfg.rounds=3;
cfg.refLevels=[3 5 7 9];
cfg.refReserve=9;
cfg.numRepeats=30;
cfg.L=3;
cfg.Ns=2000;
cfg.numTrajectories=10;
cfg.etaSigma=0.5;
cfg.xi=0.4;
cfg.epsilonAssignment=1e-6;
cfg.epsilonPhi=1e-5;
cfg.splitSeed=20261009;
cfg.chainSeed=271828;
cfg.feedbackSeed=314159;
cfg.bootstrapSeed=161803;
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
cfg.verbose=false;
cfg.displayDetails=false;
cfg.progressEvery=1000;
cfg.queue=[];
cfg.progressQueue=[];
cfg.progressMethod='AVG-TP-TC';
cfg.logProgress=true;
cfg.datasets={'CEV','LEV','MMG'};
cfg.dataDir=fullfile(cfg.scriptDir,'Datasets');
cfg.colors=[169 215 240;48 170 208;253 192 27;255 117 4]/255;
assert(exist('linprog','file')==2,'Optimization Toolbox is required.');
assert(exist('parpool','file')==2,'Parallel Computing Toolbox is required.');
assert(mod(cfg.Ns,cfg.nChains)==0 && mod(cfg.chainDraws,cfg.Ns/cfg.nChains)==0);
assert(cfg.poolCount>=cfg.rounds*cfg.batchSize);
assert(all(cfg.refLevels<=cfg.refReserve));
oldPath=path; pathGuard=onCleanup(@()path(oldPath));
oldRng=rng; rngGuard=onCleanup(@()rng(oldRng));
addpath(cfg.scriptDir);
cfg.useMexSampler=exist('har_chord_mex','file')==3;
nD=numel(cfg.datasets); dataSets=cell(nD,1); configs=cell(nD,1);
for d=1:nD
    localCfg=cfg; localCfg.dataset=cfg.datasets{d}; localCfg.datasetID=d;
    [X,y,readInfo]=readData(fullfile(cfg.dataDir,[localCfg.dataset,'.csv']));
    localCfg.numObjects=size(X,1); localCfg.numCriteria=size(X,2); localCfg.nClasses=max(y);
    localCfg.classCounts=accumarray(y,1,[localCfg.nClasses,1]);
    localCfg.rescaledColumns=find(readInfo.Transformed);
    assert(all(localCfg.classCounts>=cfg.refReserve+2), ...
        '%s: insufficient class objects for references, pool and test.',localCfg.dataset);
    assert(localCfg.numObjects-cfg.refReserve*localCfg.nClasses-cfg.poolCount>=localCfg.nClasses, ...
        '%s: insufficient common test objects.',localCfg.dataset);
    dataSets{d}=struct('X',X,'y',y); configs{d}=localCfg;
    probe=makeSplit(X,y,1,localCfg);
    reference=numel(probe.Reservoir);
    fprintf('%s: N=%d, classes=%d, reference=%d, initial pool=%d, fixed test=%d.\n', ...
        localCfg.dataset,localCfg.numObjects,localCfg.nClasses,reference,numel(probe.Pool), ...
        localCfg.numObjects-reference-numel(probe.Pool));
end
totalRounds=nD*cfg.numRepeats*numel(cfg.refLevels)*cfg.rounds;
expected=nD*cfg.numRepeats*numel(cfg.refLevels)*(cfg.rounds+1);
pool=gcp('nocreate');
if ~isempty(pool) && pool.NumWorkers~=cfg.workers, delete(pool); pool=[]; end
if isempty(pool)
    cluster=parcluster('local'); cluster.NumWorkers=max(cluster.NumWorkers,cfg.workers);
    evalc('pool=parpool(cluster,cfg.workers);');
end
assert(~isa(pool,'parallel.ThreadPool'),'Use a process-based pool.');
addAttachedFiles(pool,{[mfilename('fullpath'),'.m']});
if cfg.useMexSampler, addAttachedFiles(pool,{which('har_chord_mex')}); end
queue=parallel.pool.DataQueue; afterEach(queue,@receivePacket);
for d=1:nD
    localCfg=configs{d}; localCfg.progressQueue=queue; configs{d}=localCfg;
end
receivePacket(struct('Kind','init','Total',totalRounds,'Expected',expected));
jobsPerDataset=cfg.numRepeats*numel(cfg.refLevels);
nJobs=nD*jobsPerDataset;
parfor j=1:nJobs
    d=floor((j-1)/jobsPerDataset)+1;
    localJob=mod(j-1,jobsPerDataset);
    repeat=floor(localJob/numel(cfg.refLevels))+1;
    level=mod(localJob,numel(cfg.refLevels))+1;
    data=dataSets{d};
    runCondition(data.X,data.y,repeat,level,configs{d});
end
drainClock=tic;
while receivePacket(struct('Kind','count'))<expected && toc(drainClock)<60
    drawnow; pause(0.05);
end
assert(receivePacket(struct('Kind','count'))==expected,'Missing queued observations.');
T=receivePacket(struct('Kind','data'));
summary=cell(nD,1);
for d=1:nD
    subset=T(T.DatasetID==d,:);
    validateResults(subset,configs{d});
    summary{d}=summarizeEfficiency(subset,configs{d});
end
writeEfficiencyTable(summary,cfg);
plotBudgetPerformance(summary,cfg);
receivePacket(struct('Kind','finish'));
end

%% Proportional Pool and Nested Reference Sets
function data=makeSplit(X,y,r,cfg)
stream=RandStream('mt19937ar','Seed',keySeed(cfg.splitSeed,cfg.datasetID,r,'classwise_nested_sets'));
maxRef=cfg.refReserve; remaining=cfg.classCounts-maxRef;
allocation=boundedAllocate(cfg.poolCount,remaining,ones(cfg.nClasses,1),remaining-1);
reservoir=zeros(maxRef,cfg.nClasses); poolIDs=[];
for h=1:cfg.nClasses
    ids=find(y==h); ids=ids(randperm(stream,numel(ids)));
    reservoir(:,h)=ids(1:maxRef);
    poolIDs=[poolIDs;ids(maxRef+(1:allocation(h)))];
end
poolIDs=sort(poolIDs);
assert(numel(poolIDs)==cfg.poolCount && numel(unique(poolIDs))==cfg.poolCount);
assert(isempty(intersect(reservoir(:),poolIDs)));
data=struct('X',X,'P',[],'y',y,'Reservoir',reservoir,'Pool',poolIDs);
end

function a=boundedAllocate(n,counts,lower,upper)
assert(sum(lower)<=n && sum(upper)>=n,'Insufficient class capacity.');
target=n*counts/sum(counts); a=min(upper,max(lower,floor(target)));
while sum(a)<n
    score=target-a; score(a>=upper)=-inf; [~,j]=max(score); a(j)=a(j)+1;
end
while sum(a)>n
    score=a-target; score(a<=lower)=-inf; [~,j]=max(score); a(j)=a(j)-1;
end
end

%% One Method, Nested References, Paired Budget Checkpoints
function runCondition(X,y,r,level,cfg)
oldThreads=maxNumCompThreads(1); threadGuard=onCleanup(@()maxNumCompThreads(oldThreads));
data=makeSplit(X,y,r,cfg); nref=cfg.refLevels(level);
ref=sort(reshape(data.Reservoir(1:nref,:),[],1));
maxRef=sort(data.Reservoir(:));
test=setdiff((1:cfg.numObjects)',[ref;data.Pool],'stable');
commonTest=setdiff((1:cfg.numObjects)',[maxRef;data.Pool],'stable');
assert(isempty(intersect(ref,data.Pool)) && isempty(intersect(ref,test)));
assert(numel(commonTest)==cfg.numObjects-cfg.refReserve*cfg.nClasses-cfg.poolCount);
prepCPU=cputime; prepClock=tic;
data.P=basis(data.X,cfg.L); context0=makeContext(data,r,cfg);
preprocessSeconds=toc(prepClock); preprocessCPUSeconds=cputime-prepCPU;
[context,cache,order]=freshCaches(context0);
reference=ref; labels=data.y(ref); active=data.Pool;
initialCPU=cputime; initialClock=tic;
[base,order]=getState(reference,labels,context,cfg,cache,order,true,'none');
currentError=predictionError(mean(base.Models,2),data.P(test,:),data.y(test),base.Region);
currentCommonError=predictionError(mean(base.Models,2),data.P(commonTest,:),data.y(commonTest),base.Region);
initialSeconds=toc(initialClock); initialCPUSeconds=cputime-initialCPU;
initialError=currentError; initialCommonError=currentCommonError;
previous=emptyRound(); selectionTotal=0; selectionCPUTotal=0; updateTotal=0; updateCPUTotal=0;
for stateIndex=0:cfg.rounds
    stateCPU=cputime; stateClock=tic;
    base=ensureEvidence(base,'TP',context,cfg); cache(base.StateKey)=base;
    state=mainState(base,active,context,cfg);
    stateSeconds=toc(stateClock); stateCPUSeconds=cputime-stateCPU;
    meta=context.ModelCache('__meta__');
    row=struct('Dataset',string(cfg.dataset),'DatasetID',cfg.datasetID,'Repeat',r,'RefPerClass',nref,'Budget',stateIndex*cfg.batchSize, ...
        'OMAE',currentError,'CommonOMAE',currentCommonError, ...
        'InitialOMAE',initialError,'InitialCommonOMAE',initialCommonError, ...
        'POSCount',sum(state.Action==1),'BNDCount',sum(state.Action==2), ...
        'NEGCount',sum(state.Action==3),'PreprocessSeconds',preprocessSeconds, ...
        'InitialSeconds',initialSeconds,'PreprocessCPUSeconds',preprocessCPUSeconds, ...
        'InitialCPUSeconds',initialCPUSeconds, ...
        'TotalSeconds',preprocessSeconds+initialSeconds+selectionTotal+updateTotal, ...
        'TotalCPUSeconds',preprocessCPUSeconds+initialCPUSeconds+selectionCPUTotal+updateCPUTotal, ...
        'RealFits',meta.RealFits,'HypotheticalFits',meta.HypotheticalFits, ...
        'InitialReferenceIDs',string(sprintf('%d;',ref)), ...
        'InitialPoolIDs',string(sprintf('%d;',data.Pool)), ...
        'TestSize',numel(test),'CommonTestSize',numel(commonTest), ...
        'WorkerPID',feature('getpid'),'Protocol',string(cfg.protocol));
    fields=fieldnames(previous);
    for f=1:numel(fields), row.(fields{f})=previous.(fields{f}); end
    send(cfg.progressQueue,struct('Kind','observation','Row',struct2table(row)));
    if stateIndex==cfg.rounds, break; end
    previous=emptyRound(); before=context.ModelCache('__meta__');
    selectionCPU=cputime; selectionClock=tic;
    [query,order,details]=mainBatch(state,reference,labels,active,context,cfg,cache,order,stateIndex+1);
    previous.SelectionSeconds=stateSeconds+toc(selectionClock);
    previous.SelectionCPUSeconds=stateCPUSeconds+cputime-selectionCPU;
    after=context.ModelCache('__meta__');
    assert(numel(query)==cfg.batchSize && numel(unique(query))==cfg.batchSize);
    assert(all(ismember(query,active)) && ~any(ismember(query,test)));
    [found,queryRows]=ismember(query,active); assert(all(found));
    previous.QueryIDs=string(sprintf('%d;',query));
    previous.QueryFromPOS=sum(state.Action(queryRows)==1);
    previous.QueryFromBND=sum(state.Action(queryRows)==2);
    previous.QueryFromNEG=sum(state.Action(queryRows)==3);
    previous.UsedSecondary=double(any(state.Action(queryRows)==2));
    previous.UsedRetention=double(any(state.Action(queryRows)==1));
    previous.OriginalNEGShort=double(sum(state.Action==3)<cfg.batchSize);
    previous.Fallback=double(details.Fallback);
    previous.TrajectorySeconds=details.TrajectorySeconds;
    previous.TrajectoryUnionSize=details.UnionSize;
    previous.SelectionHypotheticalFits=after.HypotheticalFits-before.HypotheticalFits;
    updateCPU=cputime; updateClock=tic;
    reference=[reference;query]; labels=[labels;data.y(query)];
    [reference,ix]=sort(reference); labels=labels(ix); active=setdiff(active,query,'stable');
    [base,order]=getState(reference,labels,context,cfg,cache,order,true,'none');
    currentError=predictionError(mean(base.Models,2),data.P(test,:),data.y(test),base.Region);
    currentCommonError=predictionError(mean(base.Models,2),data.P(commonTest,:),data.y(commonTest),base.Region);
    previous.UpdateSeconds=toc(updateClock); previous.UpdateCPUSeconds=cputime-updateCPU;
    selectionTotal=selectionTotal+previous.SelectionSeconds;
    selectionCPUTotal=selectionCPUTotal+previous.SelectionCPUSeconds;
    updateTotal=updateTotal+previous.UpdateSeconds;
    updateCPUTotal=updateCPUTotal+previous.UpdateCPUSeconds;
end
end

function out=emptyRound()
out=struct('QueryIDs',string('-'),'SelectionSeconds',0,'SelectionCPUSeconds',0, ...
    'UpdateSeconds',0,'UpdateCPUSeconds',0,'TrajectorySeconds',0,'TrajectoryUnionSize',NaN, ...
    'SelectionHypotheticalFits',0,'QueryFromPOS',0,'QueryFromBND',0,'QueryFromNEG',0, ...
    'UsedSecondary',0,'UsedRetention',0,'OriginalNEGShort',0,'Fallback',0);
end

%% Shared Geometry, Independent Caches
function context=makeContext(data,r,cfg)
pool=data.Pool; X=data.X;
context=struct('DatasetID',cfg.datasetID,'Repeat',r,'X',X,'P',data.P,'q',cfg.nClasses, ...
    'Universe',pool,'nCriteria',cfg.numCriteria);
D=sqrt(max(0,sum((reshape(X(pool,:),numel(pool),1,[])- ...
    reshape(X(pool,:),1,numel(pool),[])).^2,3)/size(X,2)));
positive=D(triu(true(numel(pool)),1)&D>0);
if isempty(positive), sigma=eps; else, sigma=median(positive); end
context.Kcov=exp(-(D/sigma).^2); context.SigmaCov=sigma;
end

function [context,cache,order]=freshCaches(common)
context=common;
context.ModelCache=containers.Map('KeyType','char','ValueType','any');
context.ModelCache('__meta__')=struct('Order',{{}},'Bytes',0,'Hits',0, ...
    'RealFits',0,'HypotheticalFits',0,'ChainTransitions',0);
cache=containers.Map('KeyType','char','ValueType','any'); order={};
end

function result=receivePacket(packet)
persistent count completed total expected clock lastPercent rows
result=[];
if strcmp(packet.Kind,'init')
    count=0; completed=0; total=packet.Total; expected=packet.Expected;
    rows=cell(expected,1); clock=tic; lastPercent=-1;
    chainProgress('reset');
    fprintf('Progress: 0%% | Elapsed: 00:00:00 | Remaining: --:--:--\n');
elseif strcmp(packet.Kind,'transition')
    chain=chainProgress('update',packet);
    if chain.changed && completed>0
        elapsed=toc(clock);
        fprintf('Progress: %d%% | Chain: %d%% | Elapsed: %s | Remaining: %s\n', ...
            max(0,lastPercent),chain.percent,durationText(elapsed),durationText(elapsed*(total-completed)/completed));
    elseif chain.changed
        fprintf('Progress: 0%% | Chain: %d%% | Elapsed: %s | Remaining: --:--:--\n', ...
            chain.percent,durationText(toc(clock)));
    end
elseif strcmp(packet.Kind,'observation')
    count=count+1; assert(count<=expected,'Too many result rows.');
    rows{count}=table2struct(packet.Row);
    if packet.Row.Budget>0
        completed=completed+1; percent=min(99,floor(100*completed/total));
        if percent>lastPercent
            elapsed=toc(clock);
            if completed>0
                remaining=durationText(elapsed*(total-completed)/completed);
            else
                remaining='--:--:--';
            end
            fprintf('Progress: %d%% | Elapsed: %s | Remaining: %s\n', ...
                percent,durationText(elapsed),remaining);
            lastPercent=percent;
        end
    end
elseif strcmp(packet.Kind,'count')
    result=count;
elseif strcmp(packet.Kind,'data')
    assert(count==expected,'Incomplete in-memory results.');
    result=struct2table(vertcat(rows{1:count}));
elseif strcmp(packet.Kind,'finish')
    assert(completed==total && count==expected,'Incomplete feedback rounds.');
    fprintf('Progress: 100%% | Elapsed: %s | Remaining: 00:00:00\n',durationText(toc(clock)));
    rows={};
end
end

function text=durationText(seconds)
seconds=max(0,round(seconds)); hours=floor(seconds/3600); minutes=floor(mod(seconds,3600)/60);
text=sprintf('%02d:%02d:%02d',hours,minutes,mod(seconds,60));
end

function text=columnText(columns)
if isempty(columns), text=string('-'); else, text=string(sprintf('%d;',columns)); end
end

%% Result Validation
function validateResults(T,cfg)
assert(height(T)==cfg.numRepeats*numel(cfg.refLevels)*(cfg.rounds+1),'Incomplete results.');
assert(height(unique(T(:,{'Repeat','RefPerClass','Budget'})))==height(T),'Duplicate observations.');
assert(all(T.Protocol==string(cfg.protocol)),'Result protocol mismatch.');
assert(all(T.DatasetID==cfg.datasetID) && all(T.Dataset==string(cfg.dataset)),'Dataset mismatch.');
assert(all(isfinite(T.CommonOMAE)&T.CommonOMAE>=0&T.CommonOMAE<=1),'Invalid common test error.');
assert(all(isfinite(T.TotalSeconds)&T.TotalSeconds>=0),'Invalid running time.');
assert(all(T.CommonTestSize==cfg.numObjects-cfg.refReserve*cfg.nClasses-cfg.poolCount));
for r=1:cfg.numRepeats
    pair=T(T.Repeat==r,:);
    assert(numel(unique(pair.InitialPoolIDs))==1,'Query pools differ across reference levels.');
    allRefs=cell(numel(cfg.refLevels),1);
    for l=1:numel(cfg.refLevels)
        nref=cfg.refLevels(l); sub=pair(pair.RefPerClass==nref,:);
        sub=sortrows(sub,'Budget'); allRefs{l}=parseIDs(sub.InitialReferenceIDs(1));
        assert(height(sub)==cfg.rounds+1);
        assert(isequal(sub.Budget,(0:cfg.batchSize:cfg.rounds*cfg.batchSize)'));
        assert(all(sub.POSCount+sub.BNDCount+sub.NEGCount==cfg.poolCount-sub.Budget));
        assert(all(sub.QueryFromPOS(2:end)+sub.QueryFromBND(2:end)+sub.QueryFromNEG(2:end)==cfg.batchSize));
        assert(all(abs(sub.InitialCommonOMAE-sub.CommonOMAE(1))<1e-12));
        assert(all(diff(sub.TotalSeconds)>=-1e-8));
        assert(all(diff(sub.HypotheticalFits)>=0));
        expected=sub.PreprocessSeconds(1)+sub.InitialSeconds(1)+sum(sub.SelectionSeconds+sub.UpdateSeconds);
        assert(abs(expected-sub.TotalSeconds(end))<1e-7*max(1,expected));
        for t=2:height(sub), assert(numel(parseIDs(sub.QueryIDs(t)))==cfg.batchSize); end
    end
    for l=1:numel(cfg.refLevels)-1
        assert(all(ismember(allRefs{l},allRefs{l+1})),'Reference sets are not nested.');
    end
end
end

function ids=parseIDs(s)
if ismissing(s) || s==string('-'), ids=[]; else, ids=sscanf(char(s),'%d;'); end
end

%% Feedback-Efficiency Summaries
function S=summarizeEfficiency(T,cfg)
nL=numel(cfg.refLevels); nR=cfg.numRepeats;
budgets=(0:cfg.batchSize:cfg.rounds*cfg.batchSize)'; nB=numel(budgets);
assert(isequal(budgets,[0;3;6;9]),'Expected budgets 0, 3, 6 and 9.');
E=nan(nR,nB,nL); Time=E; TCTime=E;
for l=1:nL
    for r=1:nR
        rows=sortrows(T(T.Repeat==r&T.RefPerClass==cfg.refLevels(l),:),'Budget');
        E(r,:,l)=rows.CommonOMAE';
        Time(r,:,l)=rows.TotalSeconds';
        TCTime(r,:,l)=cumsum(rows.TrajectorySeconds)';
    end
end
meanE=reshape(mean(E,1),nB,nL)';
sdE=reshape(std(E,0,1),nB,nL)';
metrics=nan(nR,8,nL); validCounts=zeros(nL,1);
for l=1:nL
    e=E(:,:,l); wall=Time(:,end,l); tc=TCTime(:,end,l);
    metrics(:,1:3,l)=-diff(e,1,2)/cfg.batchSize;
    metrics(:,4,l)=(e(:,1)-e(:,end))/budgets(end);
    valid=e(:,1)>0; validCounts(l)=sum(valid);
    rer=nan(nR,1); rer(valid)=100*(e(valid,1)-e(valid,end))./e(valid,1);
    share=nan(nR,1); positive=wall>0; share(positive)=100*tc(positive)./wall(positive);
    assert(all(tc<=wall+1e-6),'TC time exceeds total time.');
    metrics(:,5,l)=rer; metrics(:,6,l)=wall;
    metrics(:,7,l)=tc; metrics(:,8,l)=share;
end
S=struct('Dataset',cfg.dataset,'ReferenceLevels',cfg.refLevels,'Budgets',budgets, ...
    'MeanOMAE',meanE,'SDOMAE',sdE,'Metrics',metrics,'RelativeValidCount',validCounts);
end

%% One LaTeX Table
function writeEfficiencyTable(summaries,cfg)
fid=fopen(fullfile(cfg.outputDir,cfg.texName),'w');
assert(fid>=0,'Cannot write LaTeX file.');
fileGuard=onCleanup(@()fclose(fid));
texLine(fid,'% Requires: booktabs, float, makecell, graphicx, amsmath.');
texLine(fid,['% Protocol: ',cfg.protocol]);
texLine(fid,sprintf('%% %d paired repetitions per dataset and reference level; pool=%d; batch=%d; rounds=%d.', ...
    cfg.numRepeats,cfg.poolCount,cfg.batchSize,cfg.rounds));
texLine(fid,sprintf('%% Seeds: split=%d; chain=%d; feedback=%d; streams also depend on dataset ID.', ...
    cfg.splitSeed,cfg.chainSeed,cfg.feedbackSeed));
for d=1:numel(summaries)
    S=summaries{d};
    for l=1:numel(cfg.refLevels)
        texLine(fid,sprintf('%% %s, references/class=%d: valid RER repetitions=%d/%d.', ...
            S.Dataset,cfg.refLevels(l),S.RelativeValidCount(l),cfg.numRepeats));
    end
end
texLine(fid,'\begin{table}[H]');
texLine(fid,'\centering');
texLine(fid,'\caption{Feedback efficiency and computational cost of AVG--TP--TC on CEV, LEV, and MMG.}');
texLine(fid,'\label{tab:rq3_feedback_efficiency}');
texLine(fid,'\begin{threeparttable}');
texLine(fid,'\small');
texLine(fid,'\renewcommand{\arraystretch}{1.16}');
texLine(fid,'\setlength{\tabcolsep}{3pt}');
texLine(fid,'\resizebox{\linewidth}{!}{%');
texLine(fid,'\begin{tabular}{lcrrrrrrrr}');
texLine(fid,'\toprule');
texLine(fid,['Dataset & \makecell{References\\per class} & ', ...
    '\makecell{ERQ\\$0$--$3$} & \makecell{ERQ\\$3$--$6$} & ', ...
    '\makecell{ERQ\\$6$--$9$} & \makecell{ERQ\\$0$--$9$} & ', ...
    '\makecell{RER@$9$\\(\%)} & \makecell{Total wall\\time (s)} & ', ...
    '\makecell{TC\\time (s)} & \makecell{TC share\\(\%)} \\']);
texLine(fid,'\midrule');
for d=1:numel(summaries)
    S=summaries{d};
    for l=1:numel(cfg.refLevels)
        if l==1, name=S.Dataset; else, name=''; end
        cells=cell(1,8);
        for k=1:8
            if k<=4, decimals=5; else, decimals=1; end
            cells{k}=meanSDText(S.Metrics(:,k,l),decimals);
        end
        texLine(fid,sprintf('%s & %d & %s \\',name,cfg.refLevels(l),strjoin(cells,' & ')));
    end
    if d<numel(summaries), texLine(fid,'\midrule'); end
end
texLine(fid,'\bottomrule');
texLine(fid,'\end{tabular}%');
texLine(fid,'}');
texLine(fid,'\makeatletter');
texLine(fid,'\gdef\TPT@hsize{\hsize\linewidth \parindent 1em \@parboxrestore}%');
texLine(fid,'\makeatother');
texLine(fid,'\begin{tablenotes}[flushleft]');
texLine(fid,'\item[]\textit{Note:} Entries are mean $\pm$ sample standard deviation over 30 repetitions. Times are cumulative through nine queries on 16 workers.');
texLine(fid,'\end{tablenotes}');
texLine(fid,'\end{threeparttable}');
texLine(fid,'\end{table}');
end

function value=meanSDText(x,decimals)
x=x(isfinite(x));
if isempty(x), value='--'; return; end
fmt=['%.' num2str(decimals) 'f'];
value=['$' sprintf(fmt,mean(x)) ' \pm ' sprintf(fmt,std(x,0)) '$'];
end

function texLine(fid,line)
fprintf(fid,'%s\n',line);
end

%% One Row, Three Dataset Panels
function plotBudgetPerformance(summaries,cfg)
fig=figure('Visible','on','Color','w','Units','pixels','Position',[80 80 1800 540]);
layout=tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
for d=1:numel(summaries)
    S=summaries{d}; b=S.Budgets(:)';
    ax=nexttile(layout,d); hold(ax,'on');
    bars=bar(ax,1:numel(b),S.MeanOMAE','grouped','BarWidth',0.72);
    for l=1:numel(cfg.refLevels)
        bars(l).FaceColor=cfg.colors(l,:); bars(l).EdgeColor='none';
        bars(l).DisplayName=sprintf('$r_{\\mathrm{ref}}=%d$',cfg.refLevels(l));
    end
    drawnow;
    for l=1:numel(cfg.refLevels)
        y=S.MeanOMAE(l,:); sd=S.SDOMAE(l,:);
        errorbar(ax,bars(l).XEndPoints,y,min(sd,y),sd, ...
            'LineStyle','none','Marker','none','Color',[0.25 0.25 0.25], ...
            'LineWidth',2,'CapSize',5,'HandleVisibility','off');
    end
    upper=max(S.MeanOMAE(:)+S.SDOMAE(:));
    ylim(ax,[0,upper+max(0.002,0.08*upper)]);
    xlim(ax,[0.5,numel(b)+0.5]); xticks(ax,1:numel(b));
    xticklabels(ax,arrayfun(@num2str,b,'UniformOutput',false));
    ax.YGrid='on'; ax.XGrid='off'; box(ax,'on');
    ax.Layer='bottom'; ax.FontName='Arial'; ax.FontSize=30; ax.LineWidth=2;
    title(ax,sprintf('(%s) %s',char('a'+d-1),S.Dataset),'FontWeight','normal','FontSize',30);
    xlabel(ax,'Cumulative query budget','FontSize',30);
    ylabel(ax,'OMAE','FontSize',30);
    legend(ax,bars,'Location','northeast','Orientation','vertical', ...
        'FontSize',30,'Box','off','AutoUpdate','off','Interpreter','latex');
end
set(findall(fig,'Type','axes'),'Box','on','LineWidth',2);
savefig(fig,fullfile(cfg.outputDir,[cfg.figureName,'.fig']));
exportgraphics(fig,fullfile(figureDirs(cfg.outputDir),'Figure4.pdf'),'ContentType','vector');
close(fig);
end

%% Dataset Reader
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

%% Test-Set Error
function e=predictionError(model,P,y,R)
pred=assignClasses(P*model(1:R.p),model(R.p+1:R.nVar),R.q);
e=mean(abs(pred-y))/(R.q-1);
end

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
    'Action',action,'Hat',hat,'RAD',base.RAD(rows),'Base',base);
end

%% mainBatch
function [query,order,info]=mainBatch(state,ref,labels,active,context,cfg,cache,order,roundID)
k=min(cfg.batchSize,numel(active));
info=struct('Fallback',sum(state.Action==3)<k,'UnionSize',NaN,'TrajectorySeconds',0);
query=zeros(0,1);
for region=cfg.priorityOrder
    missing=k-numel(query);
    if missing==0, break; end
    members=find(state.Action==region);
    if isempty(members), continue; end
    switch region
        case 3
            if numel(members)>=missing
                [added,order,tcInfo]=trajectorySelect(state,ref,labels,members,context,cfg,cache,order,roundID,missing);
                info.TrajectorySeconds=tcInfo.TrajectorySeconds;
                info.UnionSize=tcInfo.UnionSize;
            else
                ranking=sortrows([-state.RAD(members),state.IDs(members)],[1,2]);
                added=ranking(:,2);
            end
        case 2
            if numel(members)<=missing
                added=sort(state.IDs(members));
            else
                [found,ix]=ismember(state.IDs(members),context.Universe);
                assert(all(found));
                weights=normalizedWeights(max(0,state.Alpha(members)-state.Score(members)));
                added=greedyCoverage(context.Kcov(ix,ix),weights,missing,state.IDs(members));
            end
        case 1
            ranking=sortrows([state.Score(members),-state.RAD(members),state.IDs(members)],[1,2,3]);
            added=ranking(1:min(missing,numel(members)),3);
        otherwise
            error('Unknown three-way region.');
    end
    query=[query;added(:)];
end
assert(numel(query)==k && numel(unique(query))==k,'Priority selection failed.');
end

function [query,order,info]=trajectorySelect(state,ref,labels,negative,context,cfg,cache,order,roundID,k)
info=struct('UnionSize',NaN,'TrajectorySeconds',0);
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
info.UnionSize=numel(universe);
query=greedyCoverage(kernel(universeRows,:),weights,k,universe);
info.TrajectorySeconds=toc(trajectoryClock);
end

%% getState
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
            'Repeat',context.Repeat,'Variant',cfg.variantLabel,'Chain',ch,'RealState',isRealState,'ReferenceCount',numel(ref));
        rng(seeds(ch),'twister'); [chain,audit]=sampleChain(R,starts(:,ch),samplingCfg);
        assert(audit.NumericalFailures==0,'Development:NumericalFailure', ...
            'Sampler numerical failures; inspect geometry. No chain shortening or resampling.');
        M(:,(ch-1)*perChain+(1:perChain))=chain';
        audits{ch}=audit;
    end
    meta=context.ModelCache('__meta__');
    if isRealState, meta.RealFits=meta.RealFits+1;
    else, meta.HypotheticalFits=meta.HypotheticalFits+1; end
    meta.ChainTransitions=meta.ChainTransitions+cfg.nChains*(cfg.burnIn+cfg.chainDraws);
    context.ModelCache('__meta__')=meta;
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

%% ensureEvidence
function base=ensureEvidence(base,kind,context,cfg)
if strcmp(kind,'none'), return; end
assert(strcmp(kind,'TP'),'This benchmark uses only TP local evidence.');
if ~isempty(base.TP), return; end
nPool=numel(context.Universe);
activeRows=find(~ismember(context.Universe,base.ReferenceIDs));
D=thresholdDistance(base.U,base.U(nPool+1:end,:),base.Models,base.Region,cfg);
base.TP=integratedEvidence(base.MS,D,base.Labels,cfg,activeRows);
end

%% modelCacheGet
function [entry,hit]=modelCacheGet(cache,key)
hit=isKey(cache,key); entry=[];
if ~hit, return; end
entry=cache(key); info=cache('__meta__');
position=find(strcmp(info.Order,key),1);
info.Order(position)=[]; info.Order{end+1}=key; info.Hits=info.Hits+1;
cache('__meta__')=info;
end

%% modelCachePut
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

%% modelCacheVictim
function victimIndex=modelCacheVictim(cache,order)
victimIndex=1;
for j=1:numel(order)
    entry=cache(order{j});
    if ~isfield(entry,'IsRealState') || ~entry.IsRealState
        victimIndex=j; return;
    end
end
end

%% modelCachePromote
function modelCachePromote(cache,key)
if ~isKey(cache,key), return; end
entry=cache(key);
if isfield(entry,'IsRealState') && entry.IsRealState, return; end
entry.IsRealState=true;
cache(key)=entry;
end

%% adequacyScore
function score=adequacyScore(p,hat,cfg)
q=size(p,2);
if strcmp(cfg.adequacyMode,'nominal') || (q==2 && strcmp(cfg.adequacyMode,'ordinal'))
    rows=(1:size(p,1))';
    score=p(sub2ind(size(p),rows,hat(:)));
    assert(all(isfinite(score)) && all(score>=-1e-12 & score<=1+1e-12));
    score=max(0,min(1,score));
    return;
end
assert(strcmp(cfg.adequacyMode,'ordinal'),'Unknown adequacy mode.');
distance=abs(hat(:)-(1:q));
if strcmp(cfg.adequacyNormalization,'recommendation_max')
    denominator=max(hat(:)-1,q-hat(:));
else, error('Use the revised recommendation_max normalization.'); end
score=1-sum(p.*distance,2)./denominator;
assert(all(score>=-1e-12&score<=1+1e-12)); score=max(0,min(1,score));
end

%% assignClasses
function labels=assignClasses(U,b,q)
labels=ones(size(U));
for h=1:q-1, labels=labels+(U>b(h)); end
end

%% classDeviation
function rho=classDeviation(U,M,R,h)
if h==1, lower=zeros(size(U)); else, lower=M(R.p+h-1,:)+R.epsilon-U; end
if h==R.q, upper=zeros(size(U)); else, upper=U-M(R.p+h,:); end
rho=max(0,max(lower,upper));
end

%% thresholdDistance
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

%% integratedEvidence
function p=integratedEvidence(ms,D,labels,cfg,activeRows)
sigma=distanceBandwidth(D(activeRows,:),cfg);
K=exp(-(D/sigma).^2); W=sum(K,2); local=zeros(size(ms));
for h=1:size(ms,2), local(:,h)=sum(K(:,labels==h),2); end
p=(ms+local)./(1+W);
assert(max(abs(sum(p,2)-1))<1e-10);
end

%% distanceBandwidth
function sigma=distanceBandwidth(D,cfg)
positive=D(D>0);
if isempty(positive), sigma=eps; else, sigma=cfg.etaSigma*median(positive); end
end

%% normalizedWeights
function w=normalizedWeights(v)
if sum(v)>0, w=v/sum(v); else, w=ones(size(v))/numel(v); end
end

%% greedyCoverage
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

%% keySeed
function seed=keySeed(base,d,r,key)
prime=2147483647;
seed=mod(double(base)+104729*d+13007*r,prime);
bytes=double(char(key));
for j=1:numel(bytes), seed=mod(seed*131+bytes(j),prime); end
seed=max(1,floor(seed));
end

%% basis
function P=basis(X,L)
P=zeros(size(X,1),size(X,2)*L);
for j=1:size(X,2)
    for k=1:L, P(:,(j-1)*L+k)=max(0,min(1,L*X(:,j)-(k-1))); end
end
end

%% makeRegion
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

%% solveLP
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
retry=optimoptions(opts,'MaxIterations',cfg.lpRetryMaxIterations);
[x,fval,flag,output]=linprog(f,As,b,Es,beq,lb,ub,retry);
res=lpPrimalResidual(x,As,b,Es,beq,lb,ub);
attempts{end+1}=struct('Algorithm',retry.Algorithm,'ExitFlag',flag, ...
    'MaxPrimalResidual',res,'Output',output);
if lpSolutionAccepted(x,fval,flag,res,cfg)
    report=lpReport(flag,output,res,retry.Algorithm,attempts);
    return;
end
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

%% lpPrimalResidual
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

%% lpSolutionAccepted
function ok=lpSolutionAccepted(x,fval,flag,res,cfg)
ok=flag>0 && ~isempty(x) && all(isfinite(x)) && ...
    isscalar(fval) && isfinite(fval) && isfinite(res) && res<=cfg.feasibilityTolerance;
end

%% lpReport
function report=lpReport(flag,output,res,algorithm,attempts)
report=struct('ExitFlag',flag,'Message',output.message, ...
    'MaxPrimalResidual',res,'Algorithm',algorithm, ...
    'Retried',numel(attempts)>1,'Attempts',{attempts});
end

%% makeStarts
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

%% preparePreconditioner
function [R,report]=preparePreconditioner(R,starts,seeds,cfg)
assert(strcmp(cfg.preconditioningMethod,'hessian'),'Only fixed Hessian geometry is supported.');
if cfg.usePreconditioning
    [R,report]=prepareHessianGeometry(R,cfg);
else
    R.DirectionBasis=R.B; report=table();
end
end

%% prepareHessianGeometry
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
end

%% sampleChain
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
hasContext=isfield(cfg,'progressContext');
hasProgress=hasContext && isfield(cfg,'logProgress') && cfg.logProgress;
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
    if hasQueue || hasProgress
        sinceReport=sinceReport+1;
        if (mod(t,progressEvery)==0 && toc(reportClock)>=15) || t==nTrans
            if hasContext
                packet=cfg.progressContext; packet.Count=sinceReport;
                packet.Transition=t; packet.TotalTransitions=nTrans;
                emitProgress(cfg,packet);
            else
                send(cfg.queue,sinceReport);
            end
            sinceReport=0; reportClock=tic;
        end
    end
end
audit=struct('Seconds',toc(t0),'Transitions',nTrans,'BurnIn',burnIn, ...
    'ContinuousDraws',chainDraws,'StoredDraws',kept,'StorageThin',keepThin,'NumericalFailures',failures,'MinChord',minWidth, ...
    'MaxChord',maxWidth,'MinStep',minStep,'MaxStructuralResidual',maxResidual, ...
    'MaxEqualityResidual',maxEq,'MaxLossResidual',maxLoss);
end

%% lineLoss
function value=lineLoss(t,lo,up,lod,upd)
value=sum(max(0,max(lo+t*lod,up+t*upd)));
end

%% affineEndpoint
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

%% bisectLine
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

%% bisectEndpoint
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

%% lossValue
function value=lossValue(x,R)
if isempty(R.y), value=0; return; end
U=R.Phi*x(1:R.p); b=x(R.p+1:end); lower=zeros(numel(R.y),1); upper=lower;
ids=R.y>1; lower(ids)=b(R.y(ids)-1)+R.epsilon-U(ids);
ids=R.y<R.q; upper(ids)=U(ids)-b(R.y(ids));
value=sum(max(0,max(lower,upper)));
end

%% residuals
function [s,e,l]=residuals(x,R)
s=max([0;R.S*x-R.sb]); e=max([0;abs(R.eq*x-R.eqValue)]); l=max(0,lossValue(x,R)-R.cap);
end

%% assertFeasible
function assertFeasible(x,R,cfg)
[s,e,l]=residuals(x,R);
assert(all(isfinite(x))&&max([s,e,l])<=cfg.feasibilityTolerance, ...
    'HAR:Feasibility','Invalid parameter: structural=%g equality=%g loss=%g',s,e,l);
end

%% appendStruct
function values=appendStruct(values,row)
if isempty(values)
    values=row;
else
    values(end+1)=orderfields(row,values(1));
end
end

function emitProgress(cfg,packet)
if ~cfg.logProgress, return; end
if isempty(cfg.progressQueue)
    progressReport(packet);
else
    send(cfg.progressQueue,packet);
end
end

function progressReport(~)
end
