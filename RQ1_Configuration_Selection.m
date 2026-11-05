clc;
clear all;

%% Settings
cfg = struct();
cfg.dataFiles = {'DBS.csv','MPG.csv','CPU.csv','ESL.csv'};
cfg.scriptDir = fileparts(mfilename('fullpath'));
if isempty(cfg.scriptDir), cfg.scriptDir = pwd; end
cfg.dataDir = fullfile(cfg.scriptDir,'Datasets');
cfg.outputDir = cfg.scriptDir;
cfg.texName = [mfilename,'.tex'];
cfg.texDigits = 4;
cfg.gainDigits = 6;

cfg.modelVariants = {'MRG','AVG','DCM'};
cfg.evidenceVariants = {'MS','VP','TP'};
cfg.batchVariants = {'TR','SC','TC'};

cfg.L = 3;
cfg.Ns = 2000;
cfg.numRepeats = 30;
cfg.numTrajectories = 10;
cfg.rounds = 3;
cfg.batchSize = 3;
cfg.refPerClass = 3;
cfg.poolCount = 50;
cfg.xi = 0.4;
cfg.etaSigma = 0.5;
cfg.coverageDistance = 'normalized_euclidean';
cfg.coverageBandwidthRule = 'initial_pool_positive_median';
cfg.boundaryCoverageRule = 'standalone_bnd';
cfg.tcCandidateDomain = 'U';
cfg.feedbackBlockSize = 10;
cfg.adequacyNormalization = 'recommendation_max';
cfg.epsilonAssignment = 1e-6;
cfg.epsilonPhi = 1e-5;

cfg.workers = 16;
cfg.useParallel = true;
cfg.splitSeed = 20261009;
cfg.chainSeed = 271828;
cfg.feedbackSeed = 314159;
cfg.bootstrapSeed = 161803;
cfg.bootstrapReplicates = 10000;
cfg.confidenceLevel = 0.95;

cfg.nChains = 4;
cfg.burnIn = 100;
cfg.chainDraws = 200000;
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
cfg.useMexSampler = true;
cfg.tieTolerance = 0;

cfg.queue = [];
cfg.verbose = false;
cfg.displayDetails = false;
cfg.progressEvery = 1000;
cfg.cacheStates = 128;
cfg.modelCacheMB = 256;

%% Data
assert(isscalar(cfg.poolCount) && isfinite(cfg.poolCount) && cfg.poolCount>=1 && cfg.poolCount==floor(cfg.poolCount));
assert(isscalar(cfg.batchSize) && isfinite(cfg.batchSize) && cfg.batchSize>=1 && cfg.batchSize==floor(cfg.batchSize));
assert(cfg.rounds==3 && cfg.numTrajectories==10);
assert(cfg.poolCount>=cfg.rounds*cfg.batchSize,'Development:Budget', ...
    'Initial pool (%d) must cover rounds x batch (%d).',cfg.poolCount,cfg.rounds*cfg.batchSize);
assert(cfg.L==3 && cfg.numRepeats>=2);
assert(isscalar(cfg.etaSigma) && isfinite(cfg.etaSigma) && cfg.etaSigma>0,'etaSigma must be positive.');
assert(cfg.xi>0 && cfg.xi<0.5 && cfg.epsilonPhi>0 && cfg.epsilonAssignment>=0);
assert(cfg.feedbackBlockSize==10);
assert(strcmp(cfg.tcCandidateDomain,'U') && strcmp(cfg.boundaryCoverageRule,'standalone_bnd'));
assert(mod(cfg.Ns,cfg.nChains)==0 && mod(cfg.chainDraws,cfg.Ns/cfg.nChains)==0);
assert(strcmp(cfg.coverageDistance,'normalized_euclidean'));
assert(strcmp(cfg.coverageBandwidthRule,'initial_pool_positive_median'));
assert(exist('linprog','file')==2,'Optimization Toolbox is required.');
if cfg.useParallel, assert(license('test','Distrib_Computing_Toolbox'),'Parallel Computing Toolbox is required.'); end
if cfg.useMexSampler
    assert(exist('har_chord_mex','file')==3,'Development:MissingMex', ...
        'Compile har_chord_mex.c with mex(''-R2018a'',''har_chord_mex.c'') before running.');
end
oldRng = rng;
rngGuard = onCleanup(@()rng(oldRng));
runLatexSelfTest(cfg);
runProtocolSelfTests(cfg);
runSelectionProtocolSelfTests(cfg);
inputs = cell(numel(cfg.dataFiles),1);
for d = 1:numel(inputs)
    file = fullfile(cfg.dataDir,cfg.dataFiles{d});
    assert(exist(file,'file')==2,'Missing CSV: %s',file);
    [X,y,preprocessing] = readData(file);
    [~,name] = fileparts(file);
    inputs{d} = struct('X',X,'y',y,'Dataset',name,'Preprocessing',preprocessing);
    saved = rng;
    rng(keySeed(cfg.splitSeed,d,1,'split'),'twister');
    probe = makeSelectionSplit(y,cfg);
    rng(saved);
    fprintf('%s: N=%d, classes=%d, reference=%d, initial pool=%d, fixed test=%d.\n', ...
        name,numel(y),max(y),numel(probe.ReferenceIDs),numel(probe.PoolIDs),numel(probe.TestIDs));
end
configs = makeConfigurations(cfg);
assert(numel(configs)==27);
if cfg.useParallel
    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers~=cfg.workers
        delete(pool);
        pool = [];
    end
    if isempty(pool)
        cluster = parcluster('local');
        cluster.NumWorkers = max(cluster.NumWorkers,cfg.workers);
        parpool(cluster,cfg.workers);
    end
end

%% Run
jobs = executeSelectionJobs(cfg,inputs,configs);
summary = summarizeSelectionJobs(jobs,configs,cfg);
results = struct('Configuration',cfg,'Configurations',configs,'Jobs',{jobs}, ...
    'AOMAE',summary.RawAOMAE,'Summary',summary);
writeSelectionLatex(summary,configs,cfg);
clear rngGuard;

%% Configurations
function configs=makeConfigurations(cfg)
configs=struct('Model',{},'Evidence',{},'Batch',{},'Name',{},'NumTrajectories',{},'CandidateDomain',{});
for m=1:3
    for e=1:3
        for b=1:3
            one=struct('Model',cfg.modelVariants{m},'Evidence',cfg.evidenceVariants{e}, ...
                'Batch',cfg.batchVariants{b},'Name', ...
                strjoin({cfg.modelVariants{m},cfg.evidenceVariants{e},cfg.batchVariants{b}},'--'), ...
                'NumTrajectories',cfg.numTrajectories,'CandidateDomain',cfg.tcCandidateDomain);
            configs(end+1)=one; %#ok<AGROW>
        end
    end
end
end

%% Experiment
function jobs=executeSelectionJobs(cfg,inputs,configs)
nTasks=numel(inputs)*cfg.numRepeats; total=nTasks*numel(configs)*cfg.rounds;
queue=parallel.pool.DataQueue; selectionProgress([],total);
afterEach(queue,@(message)selectionProgress(message,total));
jobs=cell(nTasks,1);
if cfg.useParallel
    parfor task=1:nTasks
        d=ceil(task/cfg.numRepeats); r=mod(task-1,cfg.numRepeats)+1;
        jobs{task}=runPairedRepeat(inputs{d},d,r,configs,cfg,queue);
    end
else
    for task=1:nTasks
        d=ceil(task/cfg.numRepeats); r=mod(task-1,cfg.numRepeats)+1;
        jobs{task}=runPairedRepeat(inputs{d},d,r,configs,cfg,queue);
    end
end
failed=cellfun(@(j)~j.Success,jobs);
if any(failed)
    for task=find(failed)'
        fprintf(2,'%s repeat %d: %s\n',jobs{task}.Dataset,jobs{task}.Repeat,jobs{task}.ErrorMessage);
    end
    error('Development:Incomplete','Incomplete paired runs: no partial ranking or TeX output is written.');
end
end

function selectionProgress(message,total)
persistent done timerStart lastPercent
if isempty(message)
    done=0; timerStart=tic; lastPercent=-1;
    chainProgress('reset');
    printProgress(0,0,total);
    return;
end
chain=chainProgress('update',message);
isRound=isstruct(message) && isfield(message,'Kind') && strcmp(message.Kind,'round');
if isRound, done=done+message.Count; end
if ~isRound && ~chain.changed, return; end
percent=min(100,floor(100*done/max(total,1)));
if percent==lastPercent && ~chain.changed, return; end
printProgress(toc(timerStart),done,total);
lastPercent=percent;
end

function printProgress(elapsed,done,total)
percent=min(100,floor(100*done/max(total,1)));
if done>0
    remaining=formatDuration(elapsed*(total-done)/done);
else
    remaining='--:--:--';
end
chain=chainProgress('read');
if chain.percent<0
    fprintf('Progress: %d%% | Elapsed: %s | Remaining: %s\n',percent,formatDuration(elapsed),remaining);
else
    fprintf('Progress: %d%% | Chain: %d%% | Elapsed: %s | Remaining: %s\n', ...
        percent,chain.percent,formatDuration(elapsed),remaining);
end
end

function text=formatDuration(seconds)
seconds=max(0,round(seconds));
text=sprintf('%02d:%02d:%02d',floor(seconds/3600),floor(mod(seconds,3600)/60),mod(seconds,60));
end

function job=runPairedRepeat(input,d,r,configs,cfg,queue)
job=struct('Success',false,'Dataset',input.Dataset,'DatasetID',d,'Repeat',r);
try
    splitSeed=keySeed(cfg.splitSeed,d,r,'split'); rng(splitSeed,'twister');
    split=makeSelectionSplit(input.y,cfg);
    ref=split.ReferenceIDs; pool=split.PoolIDs; test=split.TestIDs;
    % Sorting gives deterministic row order without altering membership.
    ref=sort(ref); pool=sort(pool); test=sort(test);
    X=input.X; y=input.y; q=max(y); P=basis(X,cfg.L);
    cfg.progressQueue=queue;
    context=struct('DatasetID',d,'Repeat',r,'X',X,'P',P,'q',q, ...
        'Universe',pool,'nCriteria',size(X,2));
    context.Coverage=makeFixedCoverage(X(pool,:),pool);
    cache=containers.Map('KeyType','char','ValueType','any'); cacheOrder={};
    context.ModelCache=containers.Map('KeyType','char','ValueType','any');
    context.ModelCache('__meta__')=struct('Order',{{}},'Bytes',0,'Hits',0);
    context.BatchCache=containers.Map('KeyType','char','ValueType','any');
    context.BatchCache('__hits__')=0;
    nConfig=numel(configs);
    allOMAE=zeros(nConfig,cfg.rounds); queryIDs=cell(nConfig,cfg.rounds);
    fallbackUsed=false(nConfig,cfg.rounds);
    negativeCounts=zeros(nConfig,cfg.rounds); tcDetails=cell(nConfig,cfg.rounds);
    stateSeeds=cell(nConfig,cfg.rounds+1);
    calls=0; hits=0;
    for c=1:nConfig
        caseCfg=cfg;
        caseCfg.numTrajectories=configs(c).NumTrajectories;
        distanceKind=requiredDistance(configs(c),caseCfg);
        labels=y(ref); reference=ref; active=pool;
        for t=1:cfg.rounds
            [base,cacheOrder,newFit]=getState(reference,labels,context,caseCfg,cache,cacheOrder,true,distanceKind);
            calls=calls+newFit; hits=hits+~newFit;
            stateSeeds{c,t}=base.Seeds;
            state=realizeState(base,configs(c),active,context,caseCfg);
            negativeCounts(c,t)=sum(state.Action==3);
            fallbackUsed(c,t)=negativeCounts(c,t)<min(cfg.batchSize,numel(active));
            [query,cacheOrder,nFits,nHits,tcDetails{c,t}]=selectBatch(state,reference,labels,active, ...
                configs(c),context,caseCfg,cache,cacheOrder,t);
            calls=calls+nFits; hits=hits+nHits;
            assert(numel(query)==min(cfg.batchSize,numel(active))&&numel(unique(query))==numel(query));
            assert(all(ismember(query,active))&&~any(ismember(query,test)));
            queryIDs{c,t}=query;
            reference=[reference;query]; labels=[labels;y(query)];
            [reference,order]=sort(reference); labels=labels(order);
            active=setdiff(active,query,'stable');
            [updated,cacheOrder,newFit]=getState(reference,labels,context,caseCfg,cache,cacheOrder,true,'none');
            calls=calls+newFit; hits=hits+~newFit;
            evaluation=representativeModel(updated,configs(c).Model,active,context,caseCfg);
            pred=assignClasses(P(test,:)*evaluation(1:updated.Region.p),evaluation(updated.Region.p+1:end),q);
            allOMAE(c,t)=mean(abs(pred-y(test)))/(q-1);
            stateSeeds{c,t+1}=updated.Seeds;
            send(queue,struct('Kind','round','Count',1,'Dataset',input.Dataset,'Repeat',r,'Config',c,'ConfigName',configs(c).Name,'Round',t));
        end
    end
    job.Split=split; job.ActualPoolCount=numel(pool);
    job.ReferenceIDs=ref; job.PoolIDs=pool; job.TestIDs=test;
    job.SplitSeed=splitSeed; job.FeedbackSeeds=zeros(1,cfg.rounds);
    for t=1:cfg.rounds
        job.FeedbackSeeds(t)=keySeed(cfg.feedbackSeed,d,r,sprintf('round=%d',t));
    end
    job.StateSeeds=stateSeeds; job.OMAE=allOMAE;
    job.Coverage=context.Coverage; job.AOMAE=mean(allOMAE,2); job.QueryIDs=queryIDs; job.FallbackUsed=fallbackUsed;
    job.NEGCounts=negativeCounts; job.TCDetails=tcDetails;
    job.FittedStates=calls; job.CacheHits=hits;
    cacheInfo=context.ModelCache('__meta__');
    job.ModelCacheHits=cacheInfo.Hits; job.ModelCacheBytes=cacheInfo.Bytes;
    job.BatchCacheHits=context.BatchCache('__hits__'); job.Success=true;
catch ME
    job.Success=false; job.ErrorID=ME.identifier; job.ErrorMessage=ME.message;
end
end

%% Split
function split=makeSelectionSplit(y,cfg)
% Original protocol: reference -> stratified pool -> every remaining row tests.
% splitData is unchanged from the original development/R_Run implementation.
[ref,pool,test]=splitData(y,cfg.refPerClass,cfg.poolCount);
ref=sort(ref); pool=sort(pool); test=sort(test);
q=max(y); counts=accumarray(y,1,[q,1]);
assert(numel(ref)==q*cfg.refPerClass && numel(pool)==cfg.poolCount);
assert(numel(test)==numel(y)-numel(ref)-numel(pool));
assert(isequal(sort([ref;pool;test]),(1:numel(y))'),'Development:Split', ...
    'Reference, pool and test must be disjoint and cover every sample.');
split=struct('ReferenceIDs',ref,'PoolIDs',pool,'TestIDs',test, ...
    'ActualPoolCount',numel(pool),'ClassCounts',counts, ...
    'PoolClassCounts',accumarray(y(pool),1,[q,1]), ...
    'TestClassCounts',accumarray(y(test),1,[q,1]), ...
    'Rule','per-class reference, stratified initial pool, all remaining samples fixed tests');
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

function runSelectionProtocolSelfTests(cfg)
saved=rng; guard=onCleanup(@()rng(saved)); rng(7729,'twister');
n=max(30,ceil(cfg.poolCount/3)+cfg.refPerClass+5);
y=[ones(n,1);2*ones(n+5,1);3*ones(n+10,1)];
state=rng; [ref,pool,test]=splitData(y,cfg.refPerClass,cfg.poolCount);
rng(state); actual=makeSelectionSplit(y,cfg);
assert(isequal(actual.ReferenceIDs,sort(ref)) && isequal(actual.TestIDs,sort(test)));
assert(isequal(actual.PoolIDs,sort(pool)));
assert(numel(actual.TestIDs)==numel(y)-3*cfg.refPerClass-cfg.poolCount);
assert(isequal(sort([actual.ReferenceIDs;actual.PoolIDs;actual.TestIDs]),(1:numel(y))'));
assert(all(actual.TestClassCounts>=1));
C=makeConfigurations(cfg); assert(numel(C)==27 && numel(unique({C.Name}))==27);
assert(all([C.NumTrajectories]==10) && all(strcmp({C.CandidateDomain},'U')));
seed=keySeed(cfg.feedbackSeed,1,1,'round=1'); stream=RandStream('mt19937ar','Seed',seed);
assert(isequal(nestedTrajectoryUniforms(seed,cfg.numTrajectories,cfg.batchSize,cfg.feedbackBlockSize), ...
    rand(stream,cfg.numTrajectories,cfg.batchSize)));
clear guard;
end

%% Summary
function S=summarizeSelectionJobs(jobs,C,cfg)
nD=numel(cfg.dataFiles); nR=cfg.numRepeats; nC=numel(C);
A=zeros(nD,nR,nC); seen=false(nD,nR);
for task=1:numel(jobs)
    j=jobs{task}; assert(j.Success && ~seen(j.DatasetID,j.Repeat));
    d=j.DatasetID; r=j.Repeat; seen(d,r)=true;
    assert(j.ActualPoolCount==cfg.poolCount && all(cellfun(@numel,j.QueryIDs(:))==cfg.batchSize));
    assert(max(abs(j.AOMAE-mean(j.OMAE,2)))<1e-14);
    A(d,r,:)=reshape(j.AOMAE,1,1,nC);
    for c=1:nC
        queried=vertcat(j.QueryIDs{c,:});
        assert(numel(unique(queried))==cfg.rounds*cfg.batchSize && ~any(ismember(queried,j.TestIDs)));
    end
end
assert(all(seen(:))); S=summarizeSelection(A,C,cfg); S.RawAOMAE=A;
end

%% State and batch
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
% Deviations are computed from exactly Eq. (25).
robust=zeros(numel(ids),R.q);
for h=1:R.q
    rho=classDeviation(U,M,R,h); robust(:,h)=max(rho,[],2);
end
rad=min(robust,[],2);
% Compute distances/evidence only when this configuration actually needs them.
% Final-round prediction uses the same samples without unused kernel matrices.
base=struct('Region',R,'Models',M,'IDs',ids,'U',U,'Assignment',assignment, ...
    'MS',support,'VP',[],'TP',[],'Dvp',[],'Dtp',[],'RAD',rad, ...
    'ReferenceIDs',ref,'Labels',labels,'Seeds',seeds,'StateKey',key, ...
    'LPInfo',lp,'Geometry',geometry,'Audit',{audits},'IsRealState',isRealState);
base=ensureEvidence(base,distanceKind,context,cfg);
if cache.Count>=cfg.cacheStates
    % Keep actual learning states; evict the least recently used hypothetical
    % state first. This changes only recomputation, never samples or seeds.
    victim=1;
    for j=1:numel(order)
        entry=cache(order{j});
        if ~entry.IsRealState, victim=j; break; end
    end
    remove(cache,order{victim}); order(victim)=[];
end
cache(key)=base; order{end+1}=key;
end

function kind=requiredDistance(config,cfg) %#ok<INUSD>
if strcmp(config.Evidence,'VP') || strcmp(config.Evidence,'TP'), kind=config.Evidence;
else, kind='none'; end % MS never needs a model-induced coverage distance
end

function base=ensureEvidence(base,kind,context,cfg)
if strcmp(kind,'none'), return; end
nPool=numel(context.Universe);
activeRows=find(~ismember(context.Universe,base.ReferenceIDs));
if strcmp(kind,'VP')
    if isempty(base.Dvp)
        P=context.P(base.IDs,:);
        base.Dvp=valueDistance(P,P,base.Models,base.Region,context.nCriteria,cfg.L);
    end
    if isempty(base.VP)||~isfield(base,'VPEta')||base.VPEta~=cfg.etaSigma
        base.VP=integratedEvidence(base.MS,base.Dvp(:,nPool+1:end),base.Labels,cfg,activeRows);
        base.VPEta=cfg.etaSigma;
    end
elseif strcmp(kind,'TP')
    if isempty(base.Dtp)
        base.Dtp=thresholdDistance(base.U,base.U,base.Models,base.Region,cfg);
    end
    if isempty(base.TP)||~isfield(base,'TPEta')||base.TPEta~=cfg.etaSigma
        base.TP=integratedEvidence(base.MS,base.Dtp(:,nPool+1:end),base.Labels,cfg,activeRows);
        base.TPEta=cfg.etaSigma;
    end
end
end

function [entry,hit]=modelCacheGet(cache,key)
hit=isKey(cache,key); entry=[];
if ~hit, return; end
entry=cache(key); info=cache('__meta__');
position=find(strcmp(info.Order,key),1);
info.Order(position)=[]; info.Order{end+1}=key; info.Hits=info.Hits+1;
cache('__meta__')=info;
end

function modelCachePut(cache,key,entry,budget)
if budget<=0, return; end
measurement=whos('entry'); entry.CacheBytes=measurement.bytes+64;
if entry.CacheBytes>budget, return; end
info=cache('__meta__');
while ~isempty(info.Order) && info.Bytes+entry.CacheBytes>budget
    % One-shot hypothetical posteriors are evicted before reusable real states.
    victimIndex=modelCacheVictim(cache,info.Order);
    victim=info.Order{victimIndex}; previous=cache(victim);
    info.Bytes=info.Bytes-previous.CacheBytes; remove(cache,victim); info.Order(victimIndex)=[];
end
cache(key)=entry; info.Order{end+1}=key; info.Bytes=info.Bytes+entry.CacheBytes;
cache('__meta__')=info;
end

function victimIndex=modelCacheVictim(cache,order)
victimIndex=1;
for j=1:numel(order)
    entry=cache(order{j});
    if ~isfield(entry,'IsRealState') || ~entry.IsRealState
        victimIndex=j; return;
    end
end
end

function modelCachePromote(cache,key)
if ~isKey(cache,key), return; end
entry=cache(key);
if isfield(entry,'IsRealState') && entry.IsRealState, return; end
entry.IsRealState=true;
cache(key)=entry;
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
alpha=((1-cfg.xi)*mac)./((1-cfg.xi)*mac+cfg.xi*(1-mac));
state=struct('IDs',active,'Rows',rows,'Model',model,'Evidence',evidence, ...
    'Score',score,'MAC',mac,'Alpha',alpha,'Action',action,'RAD',base.RAD(rows),'Base',base);
end

function model=representativeModel(base,variant,active,context,cfg)
M=base.Models; R=base.Region;
if strcmp(variant,'AVG'), model=mean(M,2); return; end
if strcmp(variant,'MRG')
    U=R.Phi*M(1:R.p,:); margins=inf(size(U));
    for i=1:numel(R.y)
        h=R.y(i);
        if h>1, margins(i,:)=min(margins(i,:),U(i,:)-M(R.p+h-1,:)-cfg.epsilonAssignment); end
        if h<R.q, margins(i,:)=min(margins(i,:),M(R.p+h,:)-U(i,:)); end
    end
    quality=min(margins,[],1);
elseif strcmp(variant,'DCM')
    % Eq. (19) unchanged: a normalized class-index-distance consensus score.
    % The revised recommendation-specific Eq. (27) does NOT change this objective.
    [found,rows]=ismember(active,base.IDs); assert(all(found));
    if isempty(rows), model=mean(M,2); return; end
    quality=zeros(1,size(M,2));
    for h=1:R.q
        quality=quality+sum(base.MS(rows,h).*(1-abs(base.Assignment(rows,:)-h)/(R.q-1)),1);
    end
    quality=quality/numel(rows);
else, error('Unknown representative variant: %s',variant);
end
candidates=find(quality==max(quality));
if numel(candidates)>1
    losses=zeros(numel(candidates),1);
    for j=1:numel(candidates), losses(j)=lossValue(M(:,candidates(j)),R); end
    candidates=candidates(losses==min(losses));
end
model=M(:,candidates(1)); assertFeasible(model,R,cfg);
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

function D=valueDistance(Pa,Pb,M,R,m,L)
% Eq. (24) simplifies exactly because theta>=0 and each criterion's basis
% is monotone: E|u_j(a)-u_j(b)| = |phi_j(a)-phi_j(b)| E(theta_j).
w=mean(M(1:R.p,:),2); D=zeros(size(Pa,1),size(Pb,1));
for j=1:m
    cols=(j-1)*L+(1:L);
    ua=Pa(:,cols)*w(cols); ub=Pb(:,cols)*w(cols);
    D=D+abs(ua-ub');
end
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

function C=makeFixedCoverage(X,ids)
% d_cov(i,j)=norm(X_i-X_j,2)/sqrt(m). All feature rows are normalized.
% Median uses UNIQUE unordered positive-distance pairs in INITIAL P_0 only.
assert(size(X,1)==numel(ids)&&size(X,2)>0&&all(isfinite(X(:))));
assert(all(X(:)>=-1e-10&X(:)<=1+1e-10),'Coverage features must be normalized to [0,1].');
n=size(X,1); m=size(X,2); D=zeros(n);
for j=1:n-1
    D(j+1:end,j)=sqrt(sum((X(j+1:end,:)-X(j,:)).^2,2)/m);
end
D=D+D'; positive=D(triu(true(n),1)&D>0);
if isempty(positive), sigma=eps; degenerate=true;
else, sigma=median(positive); degenerate=false; end
K=exp(-(D/sigma).^2);
C=struct('IDs',ids(:),'Distance',D,'Sigma',sigma,'Kernel',K, ...
    'PositivePairs',numel(positive),'Degenerate',degenerate, ...
    'Rule','normalized Euclidean; initial-pool positive-pair median; fixed per repetition');
end

function K=fixedCoverage(context,candidateIDs,demandIDs)
[fa,a]=ismember(candidateIDs,context.Coverage.IDs);
[fb,b]=ismember(demandIDs,context.Coverage.IDs);
assert(all(fa)&&all(fb),'Coverage lookup outside the initial query pool.');
K=context.Coverage.Kernel(a,b);
end

function [query,order,nFits,nHits,details]=selectBatch(state,ref,labels,active,config,context,cfg,cache,order,roundID)
k=min(cfg.batchSize,numel(active)); query=[]; nFits=0; nHits=0; details=struct('Activated',false);
if k==0, return; end
negative=find(state.Action==3);
if numel(negative)<k
    % Original standalone-BND fallback shared by all 27 configurations. Real models and regions
    % remain fixed throughout the batch: no actual/hypothetical feedback here.
    ranking=sortrows([-state.RAD(negative),state.IDs(negative)],[1,2]);
    query=ranking(:,2); % every NEG is mandatory; RAD fixes its reporting order
    boundary=find(state.Action==2);
    take=min(k-numel(query),numel(boundary));
    if take==numel(boundary) && take>0
        query=[query;sort(state.IDs(boundary))]; % no choice if all BND must fit
    elseif take>0
        K=fixedCoverage(context,state.IDs(boundary),state.IDs(boundary));
        weights=normalizedWeights(max(0,state.Alpha(boundary)-state.Score(boundary)));
        added=greedyCoverage(K,weights,take,state.IDs(boundary));
        query=[query;added];
    end
    remaining=k-numel(query);
    if remaining>0
        positive=find(state.Action==1);
        % Weakest adequacy first; then larger RAD; then smaller global ID.
        ranking=sortrows([state.Score(positive),-state.RAD(positive),state.IDs(positive)],[1,2,3]);
        assert(size(ranking,1)>=remaining,'Fallback cannot fill prescribed batch.');
        query=[query;ranking(1:remaining,3)];
    end
    assert(numel(query)==k && numel(unique(query))==k && all(ismember(query,active)));
    return;
elseif strcmp(config.Batch,'TR')
    ranking=sortrows([-state.RAD(negative),state.IDs(negative)],[1,2]);
    query=ranking(1:k,2);
elseif strcmp(config.Batch,'SC')
    demandIDs=state.IDs(negative);
    K=fixedCoverage(context,demandIDs,demandIDs);
    weights=normalizedWeights(state.RAD(negative));
    query=greedyCoverage(K,weights,k,demandIDs);
elseif strcmp(config.Batch,'TC')
    eligible=sort(state.IDs(negative));
    prefixKey=[state.Base.StateKey,'|',config.Evidence,'|fixed_euclidean|', ...
        sprintf('eta=%.17g|',cfg.etaSigma),sprintf('%d,',[roundID;k;eligible])];
    batchKey=[prefixKey,sprintf('|R=%d|domain=%s',cfg.numTrajectories,config.CandidateDomain)];
    if isKey(context.BatchCache,batchKey)
        saved=context.BatchCache(batchKey); query=saved.Query; details=saved.Details;
        context.BatchCache('__hits__')=context.BatchCache('__hits__')+1;
        return;
    end
    demandIDs=zeros(cfg.numTrajectories,k); importance=zeros(cfg.numTrajectories,k);
    hypotheticalLabels=zeros(cfg.numTrajectories,k);
    trajectorySeeds=cell(cfg.numTrajectories,k); trajectoryKeys=trajectorySeeds;
    kernel=zeros(numel(eligible),cfg.numTrajectories*k);
    feedbackSeed=keySeed(cfg.feedbackSeed,context.DatasetID,context.Repeat,sprintf('round=%d',roundID));
    uniformInputs=nestedTrajectoryUniforms(feedbackSeed,cfg.numTrajectories,k,cfg.feedbackBlockSize);
    for trajectory=1:cfg.numTrajectories
        tmpRef=ref; tmpLabels=labels; visited=[];
        for position=1:k
            [base,order,newFit]=getState(tmpRef,tmpLabels,context,cfg,cache,order,false,requiredDistance(config,cfg));
            nFits=nFits+newFit; nHits=nHits+~newFit;
            trajectorySeeds{trajectory,position}=base.Seeds; trajectoryKeys{trajectory,position}=base.StateKey;
            remaining=setdiff(eligible,visited,'stable');
            [found,rows]=ismember(remaining,base.IDs); assert(all(found));
            ranked=sortrows([-base.RAD(rows),remaining],[1,2]); chosen=ranked(1,2);
            [~,row]=ismember(chosen,base.IDs);
            demandIDs(trajectory,position)=chosen; importance(trajectory,position)=base.RAD(row);
            p=base.(config.Evidence); cumulative=cumsum(p(row,:)); cumulative(end)=1;
            hypothetical=find(uniformInputs(trajectory,position)<cumulative,1);
            hypotheticalLabels(trajectory,position)=hypothetical;
            tmpRef=[tmpRef;chosen]; tmpLabels=[tmpLabels;hypothetical]; visited=[visited;chosen];
        end
    end
    % Reconstruct kernels in exactly the original trajectory-major order.
    for trajectory=1:cfg.numTrajectories
        for position=1:k
            column=(trajectory-1)*k+position;
            kernel(:,column)=fixedCoverage(context,eligible,demandIDs(trajectory,position));
        end
    end
    universe=unique(demandIDs(:)); [~,universeRows]=ismember(universe,eligible);
    weights=zeros(cfg.numTrajectories*k,1);
    for trajectory=1:cfg.numTrajectories
        weights((trajectory-1)*k+(1:k))=normalizedWeights(importance(trajectory,:)');
    end
    assert(strcmp(config.CandidateDomain,'U'));
    query=greedyCoverage(kernel(universeRows,:),weights,k,universe);
    % Retain only the records needed to reproduce stochastic query generation.
    details=struct('Activated',true,'R',cfg.numTrajectories, ...
        'DemandIDs',demandIDs,'HypotheticalLabels',hypotheticalLabels, ...
        'UniformInputs',uniformInputs,'FeedbackSeed',feedbackSeed, ...
        'HypotheticalStateSeeds',{trajectorySeeds},'HypotheticalStateKeys',{trajectoryKeys});
    context.BatchCache(batchKey)=struct('Query',query,'Details',details);

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

function inputs=nestedTrajectoryUniforms(seed,R,k,blockSize)
% Prefix invariant in R. First 10 rows exactly match rand(stream,10,k)
% from the original formal R=10 selector. Later blocks append more rows.
stream=RandStream('mt19937ar','Seed',seed); inputs=zeros(R,k);
for first=1:blockSize:R
    block=rand(stream,blockSize,k); last=min(R,first+blockSize-1);
    inputs(first:last,:)=block(1:last-first+1,:);
end
end

function S=summarizeSelection(A,configs,cfg)
assert(all(isfinite(A(:)))&&all(A(:)>=0&A(:)<=1));
[nD,nR,nC]=size(A);
S.Mean=reshape(mean(A,2),nD,nC); S.SD=reshape(std(A,0,2),nD,nC);
S.Overall=mean(S.Mean,1)';
ordered=sortrows([S.Overall,(1:nC)'],[1,2]); S.Order=ordered(:,2);
S.Rank=zeros(nC,1); S.Rank(S.Order)=(1:nC)'; S.Winner=S.Order(1);
% Equal exact scores receive average ranks in the marginal table.
S.MarginalRanks=pooledRanks(S.Overall);
stream=RandStream('mt19937ar','Seed',cfg.bootstrapSeed);
B=zeros(cfg.bootstrapReplicates,nC); frequencies=zeros(nC,1);
for b=1:cfg.bootstrapReplicates
    aggregate=zeros(1,nC);
    for d=1:nD
        ix=randi(stream,nR,nR,1);
        paired=reshape(A(d,ix,:),nR,nC);
        aggregate=aggregate+mean(paired,1)/nD;
    end
    B(b,:)=aggregate;
    % Same prespecified deterministic tie rule as observed winner selection.
    [~,winner]=min(aggregate); frequencies(winner)=frequencies(winner)+1;
end
alpha=(1-cfg.confidenceLevel)/2;
S.CI=zeros(nC,2); S.Delta=S.Overall-S.Overall(S.Winner); S.DeltaCI=zeros(nC,2);
for c=1:nC
    S.CI(c,:)=[empiricalQuantile(B(:,c),alpha),empiricalQuantile(B(:,c),1-alpha)];
    delta=B(:,c)-B(:,S.Winner);
    S.DeltaCI(c,:)=[empiricalQuantile(delta,alpha),empiricalQuantile(delta,1-alpha)];
end
S.SelectionFrequency=frequencies/cfg.bootstrapReplicates;
S.BootstrapTieRule='original configuration order, unrounded scores';
S.BootstrapOverall=B;
S.LeaveOneDatasetOutWinner=zeros(nD,1);
for d=1:nD
    other=setdiff(1:nD,d); [~,S.LeaveOneDatasetOutWinner(d)]=min(mean(S.Mean(other,:),1));
end
S.ConfigNames={configs.Name};
end

%% Tables
function writeSelectionLatex(S,C,cfg)
path=fullfile(cfg.outputDir,cfg.texName);
tempPath=[tempname(cfg.outputDir),'.tex']; fid=fopen(tempPath,'w');
assert(fid>0,'Cannot open temporary TeX output: %s',tempPath);
guard=onCleanup(@()cleanupTexFile(fid,tempPath));
eol=[char(92),char(92)];
names=cellfun(@(x)erase(x,'.csv'),cfg.dataFiles,'UniformOutput',false);
winner=S.Winner; chosen=C(winner);
fprintf(fid,'%% Requires: booktabs, float, multirow, makecell, threeparttable. Three tables; notes on the first two.\n');
fprintf(fid,'%% eta_sigma=%g; P0=%d; batch=%d; rounds=%d; budget=%d; R=%d; Ns=%d; repeats=%d.\n', ...
    cfg.etaSigma,cfg.poolCount,cfg.batchSize,cfg.rounds,cfg.batchSize*cfg.rounds, ...
    cfg.numTrajectories,cfg.Ns,cfg.numRepeats);
fprintf(fid,'%% Seeds: split=%d; chain=%d; feedback=%d; bootstrap=%d.\n', ...
    cfg.splitSeed,cfg.chainSeed,cfg.feedbackSeed,cfg.bootstrapSeed);
fprintf(fid,'%% Full sampler: %d chains x (%d burn-in + %d production draws); MEX=%d.\n', ...
    cfg.nChains,cfg.burnIn,cfg.chainDraws,cfg.useMexSampler);
fprintf(fid,'%% Winner minimizes full-precision overall AOMAE across all 27 configurations.\n');

% Table 4: dataset means, overall, intervals, paired difference, selection frequency, rank.
latexBegin(fid,'Configuration-selection AOMAE results.', ...
    'tab:configuration_results','lcccccccccr',3,0,true);
fprintf(fid,'%s\n',['Configuration & ',strjoin(names,' & '), ...
    ' & Overall & \makecell{95\% CI\\of Overall} & \makecell{Paired\\difference $\Delta$} & ', ...
    '\makecell{95\% CI\\of $\Delta$} & \makecell{Bootstrap\\selected (\%)} & Rank ',eol]);
fprintf(fid,'%s\n','\midrule');
for j=1:numel(C)
    c=S.Order(j); name=C(c).Name; overall=fmt(S.Overall(c),cfg); rank=num2str(S.Rank(c));
    selected=sprintf('%.2f',100*S.SelectionFrequency(c));
    if c==winner
        name=['\textbf{',name,'}']; overall=['\textbf{',overall,'}'];
        selected=['\textbf{',selected,'}']; rank=['\textbf{',rank,'}'];
        delta='---'; deltaCI='---';
    else
        delta=gainNumber(S.Delta(c),cfg); deltaCI=intervalText(S.DeltaCI(c,:),cfg);
    end
    cells=cell(1,numel(names));
    for d=1:numel(names), cells{d}=fmt(S.Mean(d,c),cfg); end
    fprintf(fid,'%s\n',[name,' & ',strjoin(cells,' & '),' & ',overall,' & ', ...
        intervalText(S.CI(c,:),cfg),' & ',delta,' & ',deltaCI,' & ',selected,' & ',rank,' ',eol]);
end
fprintf(fid,'%s\n','\bottomrule');
fprintf(fid,'%s\n','\end{tabular}');
fprintf(fid,'%s\n','\begin{tablenotes}[flushleft]');
fprintf(fid,'%s\n','\footnotesize');
fprintf(fid,'%s\n','\item[]\textit{Note:} Dataset entries are means over 30 paired repetitions. $\Delta$ is Overall minus the Overall of AVG--TP--TC. The intervals and the selection percentage use 10{,}000 paired bootstrap resamples.');
fprintf(fid,'%s\n','\end{tablenotes}');
fprintf(fid,'%s\n','\end{threeparttable}');
fprintf(fid,'%s\n','\end{table}');
fprintf(fid,'\n');

% Table 5: marginal AOMAE and the conditional AOMAE with the other modules fixed.
latexBegin(fid,'Marginal and conditional comparisons of configurable module variants.', ...
    'tab:module_marginal_results','lcccc',5,0,true);
fprintf(fid,'%s\n',['Module & Variant & Marginal Overall AOMAE & Conditional Overall AOMAE & ', ...
    '$\Delta$ from selected ',eol]);
fprintf(fid,'%s\n','\midrule');
modules={'Representative model','Evidence construction','Batch-query selection'};
fields={'Model','Evidence','Batch'};
variants={cfg.modelVariants,cfg.evidenceVariants,cfg.batchVariants};
for m=1:3
    marginal=zeros(3,1); conditional=zeros(3,1);
    for v=1:3
        mask=strcmp({C.(fields{m})},variants{m}{v});
        marginal(v)=mean(S.Overall(mask));
        conditional(v)=moduleConditional(C,S.Overall,chosen,fields{m},variants{m}{v});
    end
    selectedVariant=find(strcmp(variants{m},chosen.(fields{m})));
    others=setdiff((1:3)',selectedVariant); others=others(:);
    [~,otherOrder]=sortrows([conditional(others),others(:)],[1,2]);
    order=[selectedVariant;others(otherOrder(:))];
    for j=1:3
        v=order(j); left='';
        if j==1, left=['\multirow{3}{*}{',modules{m},'}']; end
        variant=variants{m}{v}; marginalText=fmt(marginal(v),cfg); conditionalText=fmt(conditional(v),cfg);
        if v==selectedVariant
            variant=['\textbf{',variant,'}'];
            marginalText=['\textbf{',marginalText,'}'];
            conditionalText=['\textbf{',conditionalText,'}'];
            delta='---';
        else
            delta=fmt(conditional(v)-S.Overall(winner),cfg);
        end
        fprintf(fid,'%s\n',[left,' & ',variant,' & ',marginalText,' & ',conditionalText,' & ',delta,' ',eol]);
    end
    if m<3, fprintf(fid,'%s\n','\midrule'); end
end
fprintf(fid,'%s\n','\bottomrule');
fprintf(fid,'%s\n','\end{tabular}');
fprintf(fid,'%s\n','\begin{tablenotes}[flushleft]');
fprintf(fid,'%s\n','\footnotesize');
fprintf(fid,'%s\n','\item[]\textit{Note:} Marginal entries average the nine configurations of the other two modules. Conditional entries fix those modules at the selected variants. $\Delta$ is the conditional Overall AOMAE minus that of AVG--TP--TC.');
fprintf(fid,'%s\n','\end{tablenotes}');
fprintf(fid,'%s\n','\end{threeparttable}');
fprintf(fid,'%s\n','\end{table}');
fprintf(fid,'\n');

% Appendix C.1: ranking after each development dataset is omitted.
latexBegin(fid,'Configuration rank after omitting one development dataset.', ...
    'tab:rq1_lodo','lccccc',5,0);
fprintf(fid,'%s\n',['\makecell{Omitted\\dataset} & Selected configuration & \makecell{Mean\\AOMAE} & ', ...
    'Runner-up configuration & \makecell{Mean\\AOMAE} & $\Delta$ ',eol]);
fprintf(fid,'%s\n','\midrule');
nD=size(S.Mean,1); nC=numel(C);
for d=1:nD
    score=mean(S.Mean(setdiff(1:nD,d),:),1)';
    ranked=sortrows([score,(1:nC)'],[1,2]);
    first=ranked(1,2); second=ranked(2,2);
    fprintf(fid,'%s\n',[names{d},' & ',C(first).Name,' & ',sprintf('%.5f',score(first)), ...
        ' & ',C(second).Name,' & ',sprintf('%.5f',score(second)), ...
        ' & ',sprintf('%.5f',score(second)-score(first)),' ',eol]);
end
fprintf(fid,'%s\n','\bottomrule');
fprintf(fid,'%s\n','\end{tabular}');
fprintf(fid,'%s\n','\end{table}');
status=fclose(fid);
assert(status==0,'Development:WriteFailed','Failed to close temporary TeX output.');
[ok,msg]=movefile(tempPath,path,'f');
assert(ok,'Development:WriteFailed','Cannot publish TeX output: %s',msg);
clear guard;
end

function cleanupTexFile(fid,path)
try, fclose(fid); catch, end
if exist(path,'file')==2, delete(path); end
end

function runLatexSelfTest(cfg)
testDir=tempname; mkdir(testDir); guard=onCleanup(@()rmdir(testDir,'s'));
testCfg=cfg; testCfg.outputDir=testDir; testCfg.texName='test.tex';
testCfg.bootstrapReplicates=10;
C=makeConfigurations(testCfg); nD=numel(testCfg.dataFiles); nC=numel(C);
A=zeros(nD,3,nC);
for c=1:nC, A(:,:,c)=0.1+c*0.001; end
S=summarizeSelection(A,C,testCfg);
writeSelectionLatex(S,C,testCfg);
t=fileread(fullfile(testDir,testCfg.texName));
assert(numel(strfind(t,'\end{table}'))==3,'Development:TeXSelfTest','Expected three complete tables.');
assert(numel(strfind(t,'\multirow{3}'))==3,'Development:TeXSelfTest','Incomplete module table.');
assert(contains(t,'tab:rq1_lodo'),'Development:TeXSelfTest','Missing leave-one-dataset-out table.');
clear guard;
end

function value=moduleConditional(C,overall,chosen,field,variant)
target=chosen; target.(field)=variant;
hit=strcmp({C.Model},target.Model) & strcmp({C.Evidence},target.Evidence) & strcmp({C.Batch},target.Batch);
assert(nnz(hit)==1,'Conditional configuration is not unique.');
value=overall(hit);
end

function value=intervalText(interval,cfg)
value=['$[',gainNumber(interval(1),cfg),',\,',gainNumber(interval(2),cfg),']$'];
end

function value=gainNumber(x,cfg)
% Keep the sign of small effects visible, including tiny CI endpoints.
if x~=0 && abs(x)<0.5*10^(-cfg.gainDigits)
    value=sprintf('%.2e',x);
else
    value=sprintf('%.*f',cfg.gainDigits,x);
end
end

function text=fmt(value,cfg)
if abs(value)<0.5*10^(-cfg.texDigits), value=0; end
text=sprintf('%.*f',cfg.texDigits,value);
end

function latexBegin(fid,caption,label,columns,padding,vspace,withNotes)
if nargin<7, withNotes=false; end
lines={sprintf('\\vspace{%.2fcm}',vspace),'\begin{table}[H]','\centering', ...
    ['\caption{',caption,'}'],['\label{',label,'}'],'\fontsize{7.8pt}{12pt}\selectfont', ...
    '\renewcommand{\arraystretch}{1.1}',sprintf('\\setlength{\\tabcolsep}{%gpt}',padding)};
if withNotes, lines{end+1}='\begin{threeparttable}'; end
lines=[lines,{['\begin{tabular}{',columns,'}'],'\toprule'}];
for j=1:numel(lines), fprintf(fid,'%s\n',lines{j}); end
end

%% Checks
function runProtocolSelfTests(cfg)
% Endpoint attainability for EVERY tentative class including interior classes.
for q=2:9
    for h=1:q
        p=zeros(1,q); p(h)=1; assert(abs(adequacyScore(p,h,cfg)-1)<1e-12);
        [~,far]=max(abs((1:q)-h)); p(:)=0; p(far)=1;
        assert(abs(adequacyScore(p,h,cfg))<1e-12);
    end
end
% Greedy coverage with diagonal kernels must select the largest demand first.
assert(isequal(greedyCoverage(eye(3),[0.1;0.7;0.2],2,[3;1;2]),[1;2]));
% Coverage geometry is computed once, without labels, model samples or eta.
coverage=makeFixedCoverage([0,0;1,0;0,0],[10;20;30]);
assert(abs(coverage.Sigma-1/sqrt(2))<1e-12&&coverage.PositivePairs==2);
assert(coverage.Kernel(1,3)==1&&abs(coverage.Kernel(1,2)-exp(-1))<1e-12);
identical=makeFixedCoverage(0.5*ones(3,2),(1:3)');
assert(identical.Degenerate&&isequal(identical.Kernel,ones(3)));
% Original standalone BND coverage: mandatory NEG does not alter its objective.
D=ones(5); D(1:6:end)=0;
D(2,3)=sqrt(-log(0.1)); D(3,2)=D(2,3);
D(1,2)=sqrt(-log(0.9)); D(2,1)=D(1,2); D(1,3)=10; D(3,1)=10;
base=struct('StateKey','fallback_probe','IDs',(1:5)','ReferenceIDs',5, ...
    'Dtp',D,'Dvp',[],'TP',ones(5,2)/2,'VP',[]);
state=struct('IDs',(1:4)','Rows',(1:4)','Base',base,'Action',[3;2;2;1], ...
    'RAD',[1;0.8;0.7;0.9],'Score',[0.1;0.25;0.25;0.9],'Alpha',[0.5;0.75;0.75;0.95]);
context=struct('Universe',(1:4)','Coverage',struct('IDs',(1:4)','Kernel',exp(-D(1:4,1:4).^2))); cache=containers.Map('KeyType','char','ValueType','any');
cache(base.StateKey)=base; order={base.StateKey}; testCfg=cfg; testCfg.batchSize=2;
for variant={'TR','SC','TC'}
    config=struct('Model','AVG','Evidence','MS','Batch',variant{1});
    [selected,~,fits,~]=selectBatch(state,[],[],state.IDs,config,context,testCfg,cache,order,1);
    assert(isequal(selected,[1;2])&&fits==0,'Standalone BND fallback is incorrect.');
end
% With no NEG, ordinary static BND coverage is retained.
zeroNeg=state; zeroNeg.Action=[1;2;2;1]; testCfg.batchSize=1;
[selected,~,fits,~]=selectBatch(zeroNeg,[],[],zeroNeg.IDs,config,context,testCfg,cache,order,1);
actualWeights=normalizedWeights(max(0,zeroNeg.Alpha(2:3)-zeroNeg.Score(2:3)));
expected=greedyCoverage(fixedCoverage(context,[2;3],[2;3]),actualWeights,1,[2;3]);
assert(isequal(selected,expected)&&fits==0);
% Insufficient BND: use all of it, then POS adequacy rather than RAD.
shortBnd=state; shortBnd.Action=[3;2;1;1]; shortBnd.Score=[0.1;0.2;0.9;0.6];
testCfg.batchSize=3;
[selected,~,fits,~]=selectBatch(shortBnd,[],[],shortBnd.IDs,config,context,testCfg,cache,order,1);
assert(isequal(selected,[1;2;4])&&fits==0);
onlyPos=state; onlyPos.Action=ones(4,1); onlyPos.Score=[0.8;0.6;0.6;0.6];
onlyPos.RAD=[1;0.1;0.9;0.9]; testCfg.batchSize=2;
[selected,~,~,~]=selectBatch(onlyPos,[],[],onlyPos.IDs,config,context,testCfg,cache,order,1);
assert(isequal(selected,[3;4]),'POS ties must use descending RAD then ascending IDs.');
% NEG sufficient: the original TR rule is retained.
fullNeg=state; fullNeg.Action=[3;3;3;1]; config.Batch='TR'; testCfg.batchSize=2;
[selected,~,~,~]=selectBatch(fullNeg,[],[],fullNeg.IDs,config,context,testCfg,cache,order,1);
assert(isequal(selected,[1;2]));

% Eq.24 shortcut checked against explicitly averaged per-model profiles.
P=[0.1,0.6;0.7,0.2]; R=struct('p',2); M=[0.2,0.8;0.8,0.2];
D=valueDistance(P,P,M,R,2,1); exact=zeros(2);
for s=1:2
    profiles=P.*M(:,s)';
    for i=1:2, for j=1:2, exact(i,j)=exact(i,j)+sum(abs(profiles(i,:)-profiles(j,:)))/2; end, end
end
assert(max(abs(D(:)-exact(:)))<1e-12);
% Bootstrap preserves EXACT paired offsets and dataset equal weights.
a=zeros(2,5,2); a(1,:,1)=0.1+(1:5)*0.001; a(2,:,1)=0.2+(1:5)*0.001; a(:,:,2)=a(:,:,1)+0.05;
c=cfg; c.bootstrapReplicates=200;
s=summarizeSelection(a,struct('Name',{'A','B'}),c);
assert(abs(s.Overall(1)-0.153)<1e-12&&max(abs(s.DeltaCI(2,:)-0.05))<1e-12);
assert(s.Winner==1&&s.SelectionFrequency(1)==1);
savedRng=rng; rng(902,'twister');
for trial=1:200
    n=3+mod(trial,25); lo=randn(n,1); up=randn(n,1);
    lod=randn(n,1); upd=randn(n,1); cap=lineLoss(0,lo,up,lod,upd)+1e-5;
    outside=(-1)^trial*(0.1+2*rand);
    if lineLoss(outside,lo,up,lod,upd)<=cap, continue; end
    expected=bisectLine(outside,lo,up,lod,upd,cap,cfg.lineIterations);
    actual=affineEndpoint(outside,lo,up,lod,upd,cap,cfg.lineIterations);
    assert(abs(expected-actual)<1e-10 && lineLoss(actual,lo,up,lod,upd)<=cap, ...
        'Affine endpoint differs from original bisection.');
end
% On-demand evidence must match the original eager computation exactly.
probeContext=struct('Universe',(1:5)','nCriteria',2,'P',basis(rand(8,2),cfg.L));
p=2*cfg.L; models=rand(p,20); models=models./sum(models,1);
models=[models;0.3*ones(1,20);0.7*ones(1,20)];
region=struct('p',p,'q',3,'epsilon',cfg.epsilonAssignment);
probe=struct('ReferenceIDs',(6:8)','IDs',(1:8)','Labels',(1:3)', ...
    'Models',models,'Region',region,'MS',ones(8,3)/3, ...
    'U',probeContext.P*models(1:p,:),'VP',[],'TP',[],'Dvp',[],'Dtp',[]);
eagerVP=valueDistance(probeContext.P,probeContext.P,models,region,2,cfg.L);
eagerTP=thresholdDistance(probe.U,probe.U,models,region,cfg);
expectedVP=integratedEvidence(probe.MS,eagerVP(:,6:8),probe.Labels,cfg,(1:5)');
expectedTP=integratedEvidence(probe.MS,eagerTP(:,6:8),probe.Labels,cfg,(1:5)');
lazy=ensureEvidence(probe,'none',probeContext,cfg);
assert(isempty(lazy.Dvp)&&isempty(lazy.Dtp));
lazy=ensureEvidence(lazy,'TP',probeContext,cfg);
lazy=ensureEvidence(lazy,'VP',probeContext,cfg);
assert(isequal(lazy.Dvp,eagerVP)&&isequal(lazy.Dtp,eagerTP));
assert(isequal(lazy.VP,expectedVP)&&isequal(lazy.TP,expectedTP));
% Reusing the same sampled-state cache across eta values must refresh evidence.
cOther=cfg; cOther.etaSigma=cfg.etaSigma/2;
refreshed=ensureEvidence(lazy,'VP',probeContext,cOther);
expectedOther=integratedEvidence(probe.MS,eagerVP(:,6:8),probe.Labels,cOther,(1:5)');
assert(isequal(refreshed.VP,expectedOther)&&refreshed.VPEta==cOther.etaSigma);
assert(isequal(refreshed.Models,lazy.Models)&&isequal(refreshed.Dvp,lazy.Dvp));
refreshedTP=ensureEvidence(lazy,'TP',probeContext,cOther);
expectedOtherTP=integratedEvidence(probe.MS,eagerTP(:,6:8),probe.Labels,cOther,(1:5)');
assert(isequal(refreshedTP.TP,expectedOtherTP)&&refreshedTP.TPEta==cOther.etaSigma);
assert(strcmp(requiredDistance(struct('Evidence','MS','Batch','SC'),cfg),'none'));


% Eviction/reuse keeps exact stored arrays; it must not mix state keys.
probeCache=containers.Map('KeyType','char','ValueType','any');
probeCache('__meta__')=struct('Order',{{}},'Bytes',0,'Hits',0);
modelCachePut(probeCache,'A',struct('Models',models),1024^2);
[entry,hit]=modelCacheGet(probeCache,'A'); assert(hit&&isequal(entry.Models,models));
modelCachePut(probeCache,'B',struct('Models',models+1),entry.CacheBytes);
[~,hit]=modelCacheGet(probeCache,'A'); assert(~hit);
[entry,hit]=modelCacheGet(probeCache,'B'); assert(hit&&isequal(entry.Models,models+1));
pin=containers.Map('KeyType','char','ValueType','any');
pin('__meta__')=struct('Order',{{}},'Bytes',0,'Hits',0);
payload=zeros(4,3);
modelCachePut(pin,'real',struct('Models',payload,'IsRealState',true),1e9);
modelCachePut(pin,'hyp',struct('Models',payload,'IsRealState',false),1e9);
realEntry=pin('real'); hypEntry=pin('hyp');
modelCachePut(pin,'extra',struct('Models',payload,'IsRealState',false), ...
    realEntry.CacheBytes+hypEntry.CacheBytes);
assert(isKey(pin,'real')&&~isKey(pin,'hyp')&&isKey(pin,'extra'));
modelCachePromote(pin,'extra'); extraEntry=pin('extra'); assert(extraEntry.IsRealState);
modelCachePromote(pin,'missing');
% The fast chain must reproduce the previous sampler on a full short trajectory.
probeX=[0.05 0.10;0.15 0.20;0.45 0.40;0.55 0.60;0.85 0.80;0.95 0.90];
probeY=[1;1;2;2;3;3];
[probeRegion,~]=makeRegion(probeX,probeY,3,cfg);
[probeStarts,~]=makeStarts(probeRegion,424242,cfg);
[probeRegion,~]=preparePreconditioner(probeRegion,probeStarts,1,cfg);
cProbe=cfg; cProbe.burnIn=3; cProbe.chainDraws=48; cProbe.Ns=12; cProbe.nChains=1;
cProbe.queue=[]; cProbe.verbose=false; cProbe.progressEvery=1000;
rng(424242,'twister'); [chainFast,auditFast]=sampleChain(probeRegion,probeStarts(:,1),cProbe);
rng(424242,'twister'); [chainRef,auditRef]=sampleChainReference(probeRegion,probeStarts(:,1),cProbe);
assert(isequal(chainFast,chainRef),'Development:SamplerDrift', ...
    'Fast sampler changed a retained draw by up to %g.',max(abs(chainFast(:)-chainRef(:))));
assert(auditFast.StoredDraws==auditRef.StoredDraws && auditFast.StoredDraws==12);
assert(auditFast.NumericalFailures==auditRef.NumericalFailures);
assert(isequal(auditFast.MinChord,auditRef.MinChord));
assert(isequal(auditFast.MaxChord,auditRef.MaxChord));
assert(isequal(auditFast.MinStep,auditRef.MinStep));
assert(isequal(auditFast.MaxStructuralResidual,auditRef.MaxStructuralResidual));
assert(isequal(auditFast.MaxEqualityResidual,auditRef.MaxEqualityResidual));
assert(isequal(auditFast.MaxLossResidual,auditRef.MaxLossResidual));
rng(savedRng);
end
%% Data and region
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
opts=optimoptions('linprog','Display','none', ...
    'ConstraintTolerance',cfg.solverTolerance,'OptimalityTolerance',cfg.solverTolerance);
[x,fval,flag,output]=linprog(f,sparse(A),b,sparse(Aeq),beq,lb,ub,opts);
assert(flag>0&&~isempty(x)&&all(isfinite(x)),'HAR:LP','linprog failed: %s',output.message);
res=max([0;A*x-b;abs(Aeq*x-beq);lb-x]);
if ~isempty(ub), res=max([res; x-ub]); end
assert(res<=cfg.feasibilityTolerance,'HAR:LPResidual','LP residual %g exceeds tolerance.',res);
report=struct('ExitFlag',flag,'Message',output.message,'MaxPrimalResidual',res);
end
%% Sampling
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
% Same hit-and-run transition as sampleChainReference. Geometry is hoisted,
% loss workspaces are reused, and each matrix-vector product is evaluated once.
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
        failures=failures+1; % keep x; recorded numerical self-loop, no redraw
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

function [chain,audit]=sampleChainReference(R,start,cfg)
% Previous sampler, retained so the startup check can require bitwise agreement.
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
    [left,right]=chord(x,d,R,cfg); width=right-left;
    assert(isfinite(width)&&width>=0,'HAR:Chord','Invalid chord.');
    minWidth=min(minWidth,width); maxWidth=max(maxWidth,width);
    if width<=64*eps(max(1,norm(x)))
        failures=failures+1;
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

function [left,right]=chord(x,d,R,cfg)
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

function ranks=pooledRanks(x)
[v,order]=sort(x(:));
first=[1;find(diff(v)~=0)+1]; last=[first(2:end)-1;numel(v)];
tiedRanks=repelem((first+last)/2,last-first+1);
ranks=zeros(size(v)); ranks(order)=tiedRanks;
ranks=reshape(ranks,size(x));
end

function q=empiricalQuantile(v,p)
v=sort(v(:)); index=1+(numel(v)-1)*p; a=floor(index); b=ceil(index);
q=v(a)+(index-a)*(v(b)-v(a));
end
%% Utilities
function values=appendStruct(values,row)
if isempty(values)
    values=row;
else
    values(end+1)=orderfields(row,values(1));
end
end
