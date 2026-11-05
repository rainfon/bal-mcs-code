function results=RQ4_Trajectory_Count
% Fixed-state trajectory simulation and paired three-round learning experiment.
% Requires MATLAB R2022a+, Optimization Toolbox, Parallel Computing Toolbox,
% Datasets/{DBS,MPG,CPU,ESL}.csv, and the compiled har_chord_mex.
% Output: RQ4_Trajectory_Count.fig, Figure7.pdf, and RQ4_Trajectory_Count.tex.
clc;
cfg=struct();
cfg.dataFiles={'DBS.csv','MPG.csv','CPU.csv','ESL.csv'};
cfg.scriptDir=fileparts(mfilename('fullpath'));
if isempty(cfg.scriptDir), cfg.scriptDir=pwd; end
cfg.dataDir=fullfile(cfg.scriptDir,'Datasets');
cfg.outputDir=cfg.scriptDir;
cfg.refPerClass=3;
cfg.L=3;
cfg.Ns=2000;
cfg.etaSigma=0.5;
cfg.xi=0.4;
cfg.epsilonAssignment=1e-6;
cfg.epsilonPhi=1e-5;
cfg.adequacyNormalization='recommendation_max';
cfg.splitSeed=2026;
cfg.chainSeed=271828;
cfg.feedbackSeed=314159;
cfg.workers=16;
cfg.useParallel=true;
cfg.nChains=4;
cfg.burnIn=100;
cfg.chainDraws=200000; % per chain; Ns/nChains retained; storage thin=400
cfg.usePreconditioning=true;
cfg.preconditioningMethod='hessian';
cfg.directionStrategy='coordinate_mixture';
cfg.sphereProbability=0.2;
cfg.solverTolerance=1e-9;
cfg.feasibilityTolerance=1e-9;
% Recovery only: normal LP calls retain the original default solver settings.
cfg.lpRetryMaxIterations=100000;
cfg.lpFallbackAlgorithm='interior-point'; % MATLAB R2022a supported algorithm
cfg.lpFallbackMaxIterations=10000;
cfg.minimumRadius=1e-12;
cfg.lineIterations=60;
cfg.barrierMaxIterations=80;
cfg.barrierNewtonTolerance=1e-9;
cfg.barrierArmijo=0.01;
cfg.barrierStepFraction=0.99;
cfg.cacheStates=64;
cfg.modelCacheMB=256;
cfg.useMexSampler=true; % required compiled har_chord_mex; same backend as development
cfg.verbose=false;
cfg.displayDetails=false;
cfg.progressEvery=1000;
cfg.queue=[];


cfg.feedbackBlockSize=10; % nested prefixes; R=10 reproduces main rand(stream,10,k)


cfg.poolValues=[30 50 70];
cfg.batchValues=[3 5];
cfg.RValues=[1 3 5 10 20 30];
cfg.numStates=30;
cfg.feedbackReplicates=20;
cfg.selectionSeedBases=314159+(0:cfg.feedbackReplicates-1)*1000003;
cfg.referenceSeedBase=987654321;
cfg.referenceR=200;
cfg.referenceCheckR=400; % second, larger reference defines final scores
cfg.referenceScoreTolerance=0.005; % flag maximum absolute normalized-score change
cfg.cvMeanFloor=1e-12;
cfg.resultName=[mfilename,'.mat'];
cfg.texName=[mfilename,'.tex'];
cfg.checkpointDir=fullfile(cfg.outputDir,'RSim_Checkpoints');
cfg.saveCheckpoints=false; cfg.resume=false;
cfg.fontSize=30; cfg.lineWidth=4; cfg.axesLineWidth=2;
cfg.errorRounds=3; cfg.errorPool=50; cfg.errorBatch=3;
cfg.errorRValues=[1 10 30]; cfg.errorSeedCount=5;
cfg.errorSeedBases=314159+(0:cfg.errorSeedCount-1)*1000003;
cfg.errorResultName='RSim_Error.mat';
cfg.errorCheckpointDir=fullfile(cfg.outputDir,'RSim_Error_Checkpoints');
assert(isequal(cfg.errorRValues,[1 10 30]) && cfg.errorRounds==3);
assert(cfg.errorPool==50 && cfg.errorBatch==3 && cfg.errorSeedCount>=2);
assert(isequal(cfg.poolValues,sort(unique(cfg.poolValues))));
assert(all(cfg.poolValues==fix(cfg.poolValues)) && min(cfg.poolValues)>=1);
assert(all(cfg.batchValues>=2 & cfg.batchValues==fix(cfg.batchValues)));
assert(issorted(cfg.RValues) && all(cfg.RValues>=1 & cfg.RValues==fix(cfg.RValues)));
assert(numel(unique(cfg.RValues))==numel(cfg.RValues));
assert(cfg.feedbackReplicates>=2 && cfg.numStates>=2);
assert(cfg.referenceCheckR>cfg.referenceR && cfg.referenceR>=max(cfg.RValues));
assert(cfg.referenceSeedBase~=cfg.feedbackSeed && ~any(cfg.selectionSeedBases==cfg.referenceSeedBase));
assert(mod(cfg.Ns,cfg.nChains)==0 && mod(cfg.chainDraws,cfg.Ns/cfg.nChains)==0);
addpath(cfg.scriptDir);
assert(exist('linprog','file')==2,'Optimization Toolbox required.');
assert(exist('har_chord_mex','file')==3,'RSim:MissingMex', ...
    'Put the compiled har_chord_mex for this MATLAB platform beside this file.');
oldRng=rng; rngGuard=onCleanup(@()rng(oldRng)); %#ok<NASGU>
simulationTests(cfg);
inputs=cell(numel(cfg.dataFiles),1);
for d=1:numel(inputs)
    file=fullfile(cfg.dataDir,cfg.dataFiles{d});
    [X,y,pre]=readData(file); [~,name]=fileparts(file);
    rng(keySeed(cfg.splitSeed,d,1,'split'),'twister');
    [ref,pool,test]=splitData(y,cfg.refPerClass,max(cfg.poolValues));
    inputs{d}=struct('Dataset',name,'X',X,'y',y,'Preprocessing',pre);
    fprintf('%s: N=%d, classes=%d, reference=%d, initial pool=%d, fixed test=%d.\n', ...
        name,numel(y),max(y),numel(ref),numel(pool),numel(test));
end
if cfg.saveCheckpoints && ~exist(cfg.checkpointDir,'dir'), mkdir(cfg.checkpointDir); end
if cfg.useParallel
    assert(license('test','Distrib_Computing_Toolbox'),'Parallel Toolbox required.');
    p=gcp('nocreate');
    if ~isempty(p) && p.NumWorkers~=cfg.workers, delete(p); p=[]; end
    if isempty(p)
        cluster=parcluster('local'); cluster.NumWorkers=max(cluster.NumWorkers,cfg.workers);
        p=parpool(cluster,cfg.workers);
    end
    addAttachedFiles(p,{which('har_chord_mex')});
    queue=parallel.pool.DataQueue;
else
    queue=[];
end
nJobs=numel(inputs)*cfg.numStates;
perPool=cfg.referenceCheckR+cfg.feedbackReplicates*max(cfg.RValues);
total=nJobs*numel(cfg.poolValues)*perPool;
progressReport([],total);
if ~isempty(queue), afterEach(queue,@(m)progressReport(m,total)); end
wall=tic; jobs=cell(nJobs,1);
if cfg.useParallel
    parfor j=1:nJobs
        d=floor((j-1)/cfg.numStates)+1; r=mod(j-1,cfg.numStates)+1;
        jobs{j}=runSimulationState(inputs{d},d,r,cfg,queue,total);
    end
else
    for j=1:nJobs
        d=floor((j-1)/cfg.numStates)+1; r=mod(j-1,cfg.numStates)+1;
        jobs{j}=runSimulationState(inputs{d},d,r,cfg,queue,total);
    end
end
[summary,referenceChecks]=summarizeSimulation(jobs,cfg);
results=struct('Configuration',cfg,'Jobs',{jobs},'Summary',summary, ...
    'ReferenceChecks',referenceChecks,'WallSeconds',toc(wall));
if any(referenceChecks.UnstableStates>0)
    warning('RSim:ReferenceUnstable', ...
        'Some 200-vs-400 reference checks exceed tolerance. Inspect ReferenceChecks; do not claim an exact reference.');
end
plotSimulation(results);
results.Downstream=runDownstreamError(inputs,cfg);
writeReportedTable(results,results.Downstream);
end

function job=runSimulationState(input,d,r,cfg,queue,total)
checkpoint=fullfile(cfg.checkpointDir,sprintf('RSim_%s_%02d.mat',input.Dataset,r));
signature=simulationSignature(input,d,r,cfg);
credit=numel(cfg.poolValues)*(cfg.referenceCheckR+cfg.feedbackReplicates*max(cfg.RValues));
if cfg.resume && exist(checkpoint,'file')==2
    saved=load(checkpoint,'signature','job');
    assert(isfield(saved,'signature') && isequaln(saved.signature,signature), ...
        'RSim:CheckpointMismatch','Checkpoint config/data differ: %s. Use another checkpoint directory.',checkpoint);
    assert(isfield(saved,'job') && saved.job.Success,'RSim:IncompleteCheckpoint','Incomplete checkpoint: %s',checkpoint);
    job=saved.job; notifyProgress(queue,total,credit); return;
end
rng(keySeed(cfg.splitSeed,d,r,'split'),'twister');
[ref,maxPool,test]=splitData(input.y,cfg.refPerClass,max(cfg.poolValues));
ref=sort(ref); maxPool=sort(maxPool); test=sort(test);
stream=RandStream('mt19937ar','Seed',keySeed(cfg.splitSeed,d,r,'nested-pool-order'));
sequence=nestedPoolOrder(maxPool,input.y,stream);
sharedModels=containers.Map('KeyType','char','ValueType','any');
sharedModels('__meta__')=struct('Order',{{}},'Bytes',0,'Hits',0);
P=basis(input.X,cfg.L); nP=numel(cfg.poolValues); poolJobs=cell(nP,1);
for pidx=1:nP
    active=sort(sequence(1:cfg.poolValues(pidx)));
    ctx=struct('DatasetID',d,'Repeat',r,'X',input.X,'P',P,'q',max(input.y), ...
        'Universe',active,'nCriteria',size(input.X,2),'ModelCache',sharedModels);
    X=input.X(active,:); n=numel(active);
    D=sqrt(max(0,sum((reshape(X,n,1,[])-reshape(X,1,n,[])).^2,3)/size(X,2)));
    pos=D(triu(true(n),1)&D>0);
    if isempty(pos), sigma=eps; else, sigma=median(pos); end
    ctx.Kcov=exp(-(D/sigma).^2);
    cache=containers.Map('KeyType','char','ValueType','any'); order={};
    localCfg=cfg; localCfg.progressQueue=queue;
    labels=input.y(ref);
    [base,order]=getState(ref,labels,ctx,localCfg,cache,order,true,'TP');
    state=mainState(base,active,ctx,localCfg);
    eligible=sort(state.IDs(state.Action==3)); nE=numel(eligible);
    part=struct('PoolCount',n,'PoolIDs',active,'NEGIDs',eligible,'NEGCount',nE, ...
        'InitialSeeds',base.Seeds,'SigmaCov',sigma,'StateKey',base.StateKey, ...
        'ReferenceBank',[],'SelectionBanks',{{}},'Conditions',{{}});
    maxK=min(max(cfg.batchValues),nE);
    if maxK<min(cfg.batchValues)
        for k=cfg.batchValues
            c=emptyCondition(k,cfg); c.Active=false;
            part.Conditions{end+1}=c;
        end
        notifyProgress(queue,total,cfg.referenceCheckR+cfg.feedbackReplicates*max(cfg.RValues));
        poolJobs{pidx}=part; continue;
    end
    [~,erows]=ismember(eligible,active); K=ctx.Kcov(erows,erows);
    refSeed=keySeed(cfg.referenceSeedBase,d,r,'reference-feedback');
    [referenceBank,order]=generateBank(ref,labels,eligible,maxK,cfg.referenceCheckR, ...
        refSeed,ctx,localCfg,cache,order,queue,total);
    banks=cell(cfg.feedbackReplicates,1);
    actualSeeds=zeros(cfg.feedbackReplicates,1);
    for s=1:cfg.feedbackReplicates
        actualSeeds(s)=keySeed(cfg.selectionSeedBases(s),d,r,'round=1');
        [banks{s},order]=generateBank(ref,labels,eligible,maxK,max(cfg.RValues), ...
            actualSeeds(s),ctx,localCfg,cache,order,queue,total);
    end
    assert(numel(unique(actualSeeds))==numel(actualSeeds) && ~any(actualSeeds==refSeed));
    part.ReferenceBank=referenceBank; part.SelectionBanks=banks;
    for k=cfg.batchValues
        c=emptyCondition(k,cfg); c.Active=nE>=k;
        if ~c.Active, part.Conditions{end+1}=c; continue; end
        w0=bankWeights(referenceBank,cfg.referenceR,k,eligible);
        w1=bankWeights(referenceBank,cfg.referenceCheckR,k,eligible);
        c.ReferenceR=cfg.referenceR; c.ReferenceCheckR=cfg.referenceCheckR;
        c.Scores=zeros(numel(cfg.RValues),cfg.feedbackReplicates);
        c.SmallReferenceScores=c.Scores; c.QueryIDs=cell(size(c.Scores));
        c.UnionSize=c.Scores; c.CV=nan(numel(cfg.RValues),1);
        c.Jaccard=c.CV; c.ScoreMean=c.CV; c.ScoreSD=c.CV;
        for a=1:numel(cfg.RValues)
            R=cfg.RValues(a);
            for s=1:cfg.feedbackReplicates
                [query,u]=selectBankBatch(banks{s},R,k,eligible,K);
                [found,qr]=ismember(query,eligible); assert(all(found));
                covered=max(K(qr,:),[],1);
                c.QueryIDs{a,s}=query; c.UnionSize(a,s)=u;
                c.Scores(a,s)=covered*w1;
                c.SmallReferenceScores(a,s)=covered*w0;
            end
            c.ScoreMean(a)=mean(c.Scores(a,:)); c.ScoreSD(a)=std(c.Scores(a,:),0,2);
            if c.ScoreMean(a)>cfg.cvMeanFloor, c.CV(a)=c.ScoreSD(a)/c.ScoreMean(a); end
            c.Jaccard(a)=pairJaccard(c.QueryIDs(a,:));
        end
        c.ReferenceMaxScoreChange=max(abs(c.Scores(:)-c.SmallReferenceScores(:)));
        c.ReferenceStable=c.ReferenceMaxScoreChange<=cfg.referenceScoreTolerance;
        smallMean=mean(c.SmallReferenceScores,2); smallSD=std(c.SmallReferenceScores,0,2);
        c.SmallReferenceCV=nan(size(smallMean)); ok=smallMean>cfg.cvMeanFloor;
        c.SmallReferenceCV(ok)=smallSD(ok)./smallMean(ok);
        c.ReferenceMaxCVChange=max(abs(c.CV-c.SmallReferenceCV),[],'omitnan');
        part.Conditions{end+1}=c;
    end
    poolJobs{pidx}=part;
end
job=struct('Dataset',input.Dataset,'DatasetID',d,'Repeat',r,'ReferenceIDs',ref, ...
    'MaximumPoolIDs',maxPool,'PoolOrder',sequence,'ReservedTestIDs',test, ...
    'Pools',{poolJobs},'Success',true);
if cfg.saveCheckpoints, writeStateCheckpoint(checkpoint,signature,job); end
end

function c=emptyCondition(k,cfg)
c=struct('BatchSize',k,'Active',false,'ReferenceR',cfg.referenceR, ...
    'ReferenceCheckR',cfg.referenceCheckR,'Scores',[],'SmallReferenceScores',[], ...
    'QueryIDs',{{}},'UnionSize',[],'CV',[],'SmallReferenceCV',[], ...
    'Jaccard',[],'ScoreMean',[],'ScoreSD',[],'ReferenceMaxScoreChange',NaN, ...
    'ReferenceMaxCVChange',NaN,'ReferenceStable',false);
end

function sequence=nestedPoolOrder(ids,y,stream)
% Class-proportional ordering of one fixed stratified maximum pool.
% True labels serve ONLY the offline experimental split, never acquisition.
q=max(y); lists=cell(q,1); sizes=zeros(q,1); taken=zeros(q,1);
for h=1:q
    one=ids(y(ids)==h); one=one(randperm(stream,numel(one)));
    lists{h}=one(:); sizes(h)=numel(one);
end
N=numel(ids); sequence=zeros(N,1);
for pos=1:N
    if pos<=q
        h=pos; assert(sizes(h)>0);
    else
        deficit=pos*sizes/N-taken; deficit(taken>=sizes)=-inf;
        [~,h]=max(deficit);
    end
    taken(h)=taken(h)+1; sequence(pos)=lists{h}(taken(h));
end
assert(isequal(sort(sequence),sort(ids)));
end

function [bank,order]=generateBank(ref,labels,eligible,k,R,seed,ctx,cfg,cache,order,queue,total)
% Each row is a complete trajectory. Hypothetical states use the unchanged
% posterior, TP integration and RAD rule from the supplied AVG-TP-TC code.
U=nestedTrajectoryUniforms(seed,R,k,cfg.feedbackBlockSize);
ids=zeros(R,k); rad=zeros(R,k); feedback=zeros(R,k);
for t=1:R
    tmpRef=ref; tmpLabels=labels; visited=[];
    for position=1:k
        [base,order]=getState(tmpRef,tmpLabels,ctx,cfg,cache,order,false,'TP');
        remaining=setdiff(eligible,visited,'stable');
        [found,rows]=ismember(remaining,base.IDs); assert(all(found));
        ranked=sortrows([-base.RAD(rows),remaining],[1,2]); chosen=ranked(1,2);
        [~,row]=ismember(chosen,base.IDs);
        ids(t,position)=chosen; rad(t,position)=base.RAD(row);
        cumulative=cumsum(base.TP(row,:)); cumulative(end)=1;
        h=find(U(t,position)<cumulative,1); feedback(t,position)=h;
        tmpRef=[tmpRef;chosen]; tmpLabels=[tmpLabels;h]; visited=[visited;chosen];
    end
    if total>0, notifyProgress(queue,total,1); end
end
bank=struct('DemandIDs',ids,'Importance',rad,'HypotheticalLabels',feedback, ...
    'UniformInputs',U,'FeedbackSeed',seed);
end

function weights=bankWeights(bank,R,k,eligible)
% Aggregation is used only for evaluation; exactly represents the weighted
% trajectory-demand objective, divided by trajectory count to lie in [0,1].
weights=zeros(numel(eligible),1);
for t=1:R
    [found,rows]=ismember(bank.DemandIDs(t,1:k),eligible); assert(all(found));
    w=normalizedWeights(bank.Importance(t,1:k)');
    weights=weights+accumarray(rows(:),w,[numel(eligible),1])/R;
end
assert(abs(sum(weights)-1)<1e-10 && all(weights>=0));
end

function [query,unionSize]=selectBankBatch(bank,R,k,eligible,K)
% Preserve original trajectory-major column ordering and visited-union domain.
demand=reshape(bank.DemandIDs(1:R,1:k)',[],1);
weights=zeros(R*k,1);
for t=1:R, weights((t-1)*k+(1:k))=normalizedWeights(bank.Importance(t,1:k)'); end
universe=unique(demand); [found,rows]=ismember(universe,eligible); assert(all(found));
[found,columns]=ismember(demand,eligible); assert(all(found));
query=greedyCoverage(K(rows,columns),weights,k,universe);
unionSize=numel(universe);
assert(numel(unique(query))==k && all(ismember(query,eligible)));
end

function notifyProgress(queue,total,count)
m=struct('Kind','round','Count',count);
if isempty(queue), progressReport(m,total); else, send(queue,m); end
end

function signature=simulationSignature(input,d,r,cfg)
fields={'poolValues','batchValues','RValues','numStates','feedbackReplicates', ...
    'selectionSeedBases','referenceSeedBase','referenceR','referenceCheckR', ...
    'referenceScoreTolerance','cvMeanFloor','L','Ns','xi','etaSigma', ...
    'refPerClass','epsilonAssignment','epsilonPhi','adequacyNormalization', ...
    'splitSeed','chainSeed','nChains','burnIn','chainDraws','usePreconditioning', ...
    'preconditioningMethod','directionStrategy','sphereProbability','solverTolerance', ...
    'feasibilityTolerance','lpRetryMaxIterations','lpFallbackAlgorithm', ...
    'lpFallbackMaxIterations','minimumRadius','lineIterations', ...
    'barrierMaxIterations','barrierNewtonTolerance','barrierArmijo', ...
    'barrierStepFraction','useMexSampler','feedbackBlockSize'};
settings=struct();
for i=1:numel(fields), settings.(fields{i})=cfg.(fields{i}); end
% Store exact data, not a weak ad hoc numeric hash.
signature=struct('CodeVersion',1,'DatasetID',d,'Repeat',r,'Settings',settings, ...
    'X',input.X,'y',input.y);
end

function writeStateCheckpoint(path,signature,job)
% Unique per state. Rename only after a successful write.
tmp=[path,'.tmp.mat']; save(tmp,'signature','job','-v7.3');
[ok,msg]=movefile(tmp,path,'f'); assert(ok,'RSim:CheckpointWrite','%s',msg);
end

function [S,C]=summarizeSimulation(jobs,cfg)
rows=struct([]); checks=struct([]);
for d=1:numel(cfg.dataFiles)
    [~,name]=fileparts(cfg.dataFiles{d});
    for p=1:numel(cfg.poolValues)
        for b=1:numel(cfg.batchValues)
            k=cfg.batchValues(b); nR=numel(cfg.RValues); n=cfg.numStates;
            cv=nan(n,nR); score=cv; jac=cv; us=cv; scoreSD=cv;
            neg=zeros(n,1); valid=false(n,1); deterministic=valid;
            changes=nan(n,1); cvChanges=changes; stable=valid;
            for r=1:n
                job=jobs{(d-1)*n+r}; part=job.Pools{p}; c=part.Conditions{b};
                neg(r)=part.NEGCount; valid(r)=c.Active;
                deterministic(r)=c.Active && part.NEGCount==k;
                if ~c.Active, continue; end
                cv(r,:)=c.CV'; score(r,:)=c.ScoreMean'; jac(r,:)=c.Jaccard';
                us(r,:)=mean(c.UnionSize,2)'; scoreSD(r,:)=c.ScoreSD';
                changes(r)=c.ReferenceMaxScoreChange; stable(r)=c.ReferenceStable;
                cvChanges(r)=c.ReferenceMaxCVChange;
            end
            check=struct('Dataset',string(name),'PoolCount',cfg.poolValues(p),'BatchSize',k, ...
                'TotalStates',n,'ActiveStates',sum(valid),'InactiveStates',sum(~valid), ...
                'ForcedStates',sum(deterministic),'MinNEG',min(neg),'MeanNEG',mean(neg), ...
                'MeanNEGperK',mean(neg)/k,'UnstableStates',sum(valid & ~stable), ...
                'MaxReferenceScoreChange',max(changes,[],'omitnan'), ...
                'MaxReferenceCVChange',max(cvChanges,[],'omitnan'));
            if isempty(checks), checks=check; else, checks(end+1)=check; end %#ok<AGROW>
            for a=1:nR
                row=struct('Dataset',string(name),'PoolCount',cfg.poolValues(p),'BatchSize',k, ...
                    'R',cfg.RValues(a),'ActiveStates',sum(valid),'CVStates',sum(isfinite(cv(:,a))), ...
                    'MeanCV',mean(cv(:,a),'omitnan'),'StateSDofCV',std(cv(:,a),0,'omitnan'), ...
                    'MeanCoverage',mean(score(:,a),'omitnan'), ...
                    'MeanSeedScoreSD',mean(scoreSD(:,a),'omitnan'), ...
                    'MeanJaccard',mean(jac(:,a),'omitnan'),'MeanUnionSize',mean(us(:,a),'omitnan'));
                if isempty(rows), rows=row; else, rows(end+1)=row; end %#ok<AGROW>
            end
        end
    end
end
S=struct2table(rows); C=struct2table(checks);
end

function plotSimulation(results)
cfg=results.Configuration; S=results.Summary;
names={'DBS','MPG','CPU','ESL'};
RValues=cfg.RValues(:)';
pools=cfg.poolValues(:)'; batches=cfg.batchValues(:)';
nCurves=numel(pools)*numel(batches);
% Manuscript palette: light blue, cyan, gold, orange. Three pools use the first, second and fourth.
colors=[169 215 240; 48 170 208; 255 117 4]/255;
styles={'-','--'}; markers={'o','s'};
fontName='Arial';
values=S.MeanUnionSize; values=values(isfinite(values));
if isempty(values), ymax=max(batches)+1; else, ymax=max(max(batches)+1,1.12*max(values)); end
fig=figure('Color','w','Name','Trajectory visited-union growth','NumberTitle','off', ...
    'Position',[20 80 2200 640],'Visible','on');
layout=tiledlayout(fig,1,4,'TileSpacing','compact','Padding','compact');
handles=gobjects(nCurves,1); labels=cell(nCurves,1);
for d=1:4
    ax=nexttile(layout); hold(ax,'on'); curve=0;
    for p=1:numel(pools)
        color=colors(mod(p-1,size(colors,1))+1,:);
        for b=1:numel(batches)
            curve=curve+1;
            z=S(S.Dataset==string(names{d}) & S.PoolCount==pools(p) & S.BatchSize==batches(b),:);
            [~,ix]=ismember(RValues,z.R); y=z.MeanUnionSize(ix);
            line=plot(ax,1:numel(RValues),y,'Color',color, ...
                'LineStyle',styles{mod(b-1,numel(styles))+1}, ...
                'Marker',markers{mod(b-1,numel(markers))+1}, ...
                'LineWidth',cfg.lineWidth,'MarkerSize',8, ...
                'MarkerFaceColor',color);
            if d==1
                handles(curve)=line;
                labels{curve}=sprintf('P_0=%d, k=%d',pools(p),batches(b));
            end
        end
    end
    title(ax,sprintf('(%c) %s','a'+d-1,names{d}),'FontWeight','normal');
    xlabel(ax,'Trajectory count R');
    if d==1, ylabel(ax,'Mean visited-union size'); end
    xticks(ax,1:numel(RValues)); xticklabels(ax,string(RValues));
    xlim(ax,[.7,numel(RValues)+.3]); ylim(ax,[0,ymax]);
    set(ax,'FontName',fontName,'FontSize',cfg.fontSize,'LineWidth',cfg.axesLineWidth, ...
        'Box','on','TickDir','out','XGrid','off','YGrid','on','GridAlpha',.15);
end
lg=legend(handles,labels,'Orientation','horizontal','NumColumns',nCurves, ...
    'Box','off','Interpreter','tex','FontName',fontName,'FontSize',cfg.fontSize);
lg.Layout.Tile='north'; drawnow;
set(fig,'Units','inches'); pos=get(fig,'Position');
set(fig,'PaperUnits','inches','PaperPosition',[0 0 pos(3) pos(4)], ...
    'PaperSize',[pos(3) pos(4)],'InvertHardcopy','off');
outDir=fileparts(mfilename('fullpath')); if isempty(outDir), outDir=pwd; end
figFile=fullfile(outDir,[mfilename,'.fig']);
set(findall(fig,'Type','axes'),'Box','on','LineWidth',2);
set(findall(fig,'-property','FontSize'),'FontSize',cfg.fontSize);
savefig(fig,figFile);
pdfFile=fullfile(figureDirs(outDir),'Figure7.pdf');
exportgraphics(fig,pdfFile,'ContentType','vector');
end

function writeReportedTable(results,downstream)
S=results.Summary; E=downstream.Summary; cfg=results.Configuration;
names={'DBS','MPG','CPU','ESL'}; reportR=cfg.errorRValues;
path=fullfile(cfg.outputDir,[mfilename,'.tex']);
fid=fopen(path,'w'); assert(fid>=0,'Cannot write %s',path);
guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
eol=[char(92),char(92)];
put(fid,'% Requires booktabs, multirow, and float.');
put(fid,'\begin{table}[H]');
put(fid,'\centering');
put(fid,'\caption{Effects of the number of trajectories on batch stability, test error, and computation time.}');
put(fid,'\label{tab:r_stability}');
put(fid,'\small');
put(fid,'\setlength{\tabcolsep}{5pt}');
put(fid,'\renewcommand{\arraystretch}{1.12}');
put(fid,'\begin{threeparttable}');
put(fid,'\begin{tabular}{llrrrr}');
put(fid,'\toprule');
put(fid,['Dataset & $R$ & Batch Jaccard $\uparrow$ & AOMAE $\downarrow$ ', ...
    '& Seed SD $\downarrow$ & Time (s) $\downarrow$ ',eol]);
put(fid,'\midrule');
for d=1:numel(names)
    for a=1:numel(reportR)
        R=reportR(a);
        z=S(S.Dataset==string(names{d}) & S.PoolCount==cfg.errorPool & ...
            S.BatchSize==cfg.errorBatch & S.R==R,:);
        e=E(E.Dataset==string(names{d}) & E.R==R,:);
        assert(height(z)==1 && height(e)==1,'Missing table result for %s, R=%d.',names{d},R);
        if a==1, label=sprintf('\\multirow{%d}{*}{%s}',numel(reportR),names{d});
        else, label=''; end
        if R==10, rlabel='\textbf{10}'; else, rlabel=num2str(R); end
        put(fid,[sprintf('%s & %s & %.4f & %.4f & %.4f & %.2f ', ...
            label,rlabel,z.MeanJaccard,e.MeanAOMAE,e.MeanSeedSD,e.MeanSeconds),eol]);
    end
    if d<numel(names), put(fid,'\midrule'); end
end
put(fid,'\bottomrule');
put(fid,'\end{tabular}');
put(fid,'\begin{tablenotes}[flushleft]');
put(fid,'\item[]\textit{Note:} $P_0=50$, $k=3$. Jaccard averages pairwise batch similarity over 20 seeds; AOMAE averages three rounds over 30 splits and 5 seeds; Seed SD is its within-split standard deviation; Time is mean wall time.');
put(fid,'\end{tablenotes}');
put(fid,'\end{threeparttable}');
put(fid,'\end{table}');
end

function results=runDownstreamError(inputs,cfg)
% Paired three-round AVG-TP-TC paths. One split is shared by every R and seed.
% Pool 50 and batch 3. Feedback seeds change hypothetical trajectory labels;
% accepted queries use the dataset labels. AOMAE averages the three test OMAEs.
if cfg.saveCheckpoints && ~exist(cfg.errorCheckpointDir,'dir')
    mkdir(cfg.errorCheckpointDir);
end
if cfg.useParallel
    assert(license('test','Distrib_Computing_Toolbox'),'Parallel Toolbox required.');
    p=gcp('nocreate');
    if ~isempty(p) && p.NumWorkers~=cfg.workers, delete(p); p=[]; end
    if isempty(p)
        cluster=parcluster('local'); cluster.NumWorkers=max(cluster.NumWorkers,cfg.workers);
        p=parpool(cluster,cfg.workers);
    end
    addAttachedFiles(p,{which('har_chord_mex')});
end
nJobs=numel(inputs)*cfg.numStates;
queue=parallel.pool.DataQueue;
afterEach(queue,@(~)progressReport(struct('Kind','round','Count',1),nJobs));
progressReport(struct('Kind','stage','Name','Downstream test error'),nJobs);
progressReport([],nJobs);
cfg.downstreamQueue=queue;
wall=tic; jobs=cell(nJobs,1);
if cfg.useParallel
    parfor j=1:nJobs
        d=floor((j-1)/cfg.numStates)+1; r=mod(j-1,cfg.numStates)+1;
        jobs{j}=runErrorState(inputs{d},d,r,cfg);
    end
else
    for j=1:nJobs
        d=floor((j-1)/cfg.numStates)+1; r=mod(j-1,cfg.numStates)+1;
        jobs{j}=runErrorState(inputs{d},d,r,cfg);
    end
end
summary=summarizeError(jobs,cfg);
results=struct('Configuration',cfg,'Jobs',{jobs},'Summary',summary,'WallSeconds',toc(wall));
end

function job=runErrorState(input,d,r,cfg)
checkpoint=fullfile(cfg.errorCheckpointDir,sprintf('RErr_%s_%02d.mat',input.Dataset,r));
signature=errorSignature(input,d,r,cfg);
if cfg.resume && exist(checkpoint,'file')==2
    saved=load(checkpoint,'signature','job');
    assert(isfield(saved,'signature') && isequaln(saved.signature,signature), ...
        'RSim:CheckpointMismatch','Error checkpoint differs: %s',checkpoint);
    job=saved.job; notifyDownstream(cfg); return;
end
rng(keySeed(cfg.splitSeed,d,r,'split'),'twister');
[ref0,pool,test]=splitData(input.y,cfg.refPerClass,cfg.errorPool);
ref0=sort(ref0); pool=sort(pool); test=sort(test);
P=basis(input.X,cfg.L); q=max(input.y); T=cfg.errorRounds;
nR=numel(cfg.errorRValues); nS=cfg.errorSeedCount;
X=input.X(pool,:); n=numel(pool);
D=sqrt(max(0,sum((reshape(X,n,1,[])-reshape(X,1,n,[])).^2,3)/size(X,2)));
pos=D(triu(true(n),1)&D>0);
if isempty(pos), sigma=eps; else, sigma=median(pos); end
omae=nan(nR,nS,T); elapsed=nan(nR,nS); queries=cell(nR,nS,T);
% Real states share one cache. Trajectory banks use a separate cache: a
% hypothetical label can repeat a real reference key while the candidate pool
% still contains that alternative. The model cache depends only on references.
for a=1:nR
    R=cfg.errorRValues(a);
    for s=1:nS
        trialClock=tic;
        models=containers.Map('KeyType','char','ValueType','any');
        models('__meta__')=struct('Order',{{}},'Bytes',0,'Hits',0);
        realCache=containers.Map('KeyType','char','ValueType','any'); realOrder={};
        ref=ref0; labels=input.y(ref); active=pool;
        ctx=struct('DatasetID',d,'Repeat',r,'X',input.X,'P',P,'q',q, ...
            'Universe',active,'PoolIDs',pool,'nCriteria',size(input.X,2), ...
            'ModelCache',models,'Kcov',exp(-(D/sigma).^2));
        local=cfg; local.progressQueue=[];
        for t=1:T
            ctx.Universe=active;
            [base,realOrder]=getState(ref,labels,ctx,local,realCache,realOrder,true,'TP');
            state=mainState(base,active,ctx,local);
            seed=keySeed(cfg.errorSeedBases(s),d,r,sprintf('round=%d',t));
            query=selectDownstreamBatch(state,ref,labels,active,ctx,local,R,seed);
            assert(numel(unique(query))==numel(query) && all(ismember(query,active)));
            assert(~any(ismember(query,test)),'Test alternative entered a query batch.');
            queries{a,s,t}=query;
            ref=[ref;query]; labels=[labels;input.y(query)]; %#ok<AGROW>
            [ref,ix]=sort(ref); labels=labels(ix); active=setdiff(active,query,'stable');
            ctx.Universe=active;
            [base,realOrder]=getState(ref,labels,ctx,local,realCache,realOrder,true,'TP');
            omae(a,s,t)=predictionError(mean(base.Models,2),P(test,:),input.y(test),base.Region);
        end
        elapsed(a,s)=toc(trialClock);
    end
end
job=struct('Dataset',input.Dataset,'DatasetID',d,'Repeat',r,'ReferenceIDs',ref0, ...
    'PoolIDs',pool,'ReservedTestIDs',test,'OMAE',omae,'AOMAE',mean(omae,3), ...
    'QueryIDs',{queries},'ElapsedSeconds',elapsed,'Success',true);
if cfg.saveCheckpoints, writeStateCheckpoint(checkpoint,signature,job); end
notifyDownstream(cfg);
end

function notifyDownstream(cfg)
if isfield(cfg,'downstreamQueue') && ~isempty(cfg.downstreamQueue)
    send(cfg.downstreamQueue,1);
end
end

function query=selectDownstreamBatch(state,ref,labels,active,ctx,cfg,R,seed)
% Kcov is indexed by the original pool. active shrinks after each round.
k=min(cfg.errorBatch,numel(active));
cache=containers.Map('KeyType','char','ValueType','any'); order={};
eligible=sort(state.IDs(state.Action==3));
if numel(eligible)>=k
    [found,rows]=ismember(eligible,ctx.PoolIDs); assert(all(found));
    [bank,order]=generateBank(ref,labels,eligible,k,R,seed,ctx,cfg,cache,order,[],0);
    [query,~]=selectBankBatch(bank,R,k,eligible,ctx.Kcov(rows,rows)); return;
end
negIDs=state.IDs(state.Action==3); query=zeros(0,1);
if ~isempty(negIDs)
    ranking=sortrows([-state.RAD(state.Action==3),negIDs],[1,2]);
    query=ranking(:,2);
end
boundary=find(state.Action==2); take=min(k-numel(query),numel(boundary));
if take>0
    [found,ix]=ismember(state.IDs(boundary),ctx.PoolIDs); assert(all(found));
    weights=normalizedWeights(max(0,state.Alpha(boundary)-state.Score(boundary)));
    added=greedyCoverage(ctx.Kcov(ix,ix),weights,take,state.IDs(boundary));
    query=[query;added]; %#ok<AGROW>
end
remaining=min(k-numel(query),sum(state.Action==1));
if remaining>0
    positive=find(state.Action==1);
    ranking=sortrows([state.Score(positive),-state.RAD(positive),state.IDs(positive)],[1,2,3]);
    query=[query;ranking(1:remaining,3)]; %#ok<AGROW>
end
assert(numel(query)==k && numel(unique(query))==k);
end

function e=predictionError(model,P,y,region)
pred=assignClasses(P*model(1:region.p),model(region.p+1:region.nVar),region.q);
e=mean(abs(pred(:)-y(:)))/(region.q-1);
end

function signature=errorSignature(input,d,r,cfg)
fields={'errorRounds','errorPool','errorBatch','errorRValues','errorSeedBases', ...
    'L','Ns','xi','etaSigma','refPerClass','epsilonAssignment','epsilonPhi', ...
    'adequacyNormalization','splitSeed','chainSeed','nChains','burnIn','chainDraws', ...
    'numStates','feedbackBlockSize','useMexSampler'};
settings=struct();
for i=1:numel(fields), settings.(fields{i})=cfg.(fields{i}); end
signature=struct('CodeVersion',1,'Kind','downstream-error','DatasetID',d,'Repeat',r, ...
    'Settings',settings,'X',input.X,'y',input.y);
end

function S=summarizeError(jobs,cfg)
names=cell(numel(cfg.dataFiles),1);
for d=1:numel(names), [~,names{d}]=fileparts(cfg.dataFiles{d}); end
n=cfg.numStates; nR=numel(cfg.errorRValues); rows=struct([]);
for d=1:numel(names)
    aomae=nan(n,nR); final=aomae; seedSD=aomae; seconds=aomae;
    for r=1:n
        job=jobs{(d-1)*n+r};
        aomae(r,:)=mean(job.AOMAE,2)';
        final(r,:)=mean(job.OMAE(:,:,cfg.errorRounds),2)';
        seedSD(r,:)=std(job.AOMAE,0,2)';
        seconds(r,:)=mean(job.ElapsedSeconds,2)';
    end
    for a=1:nR
        row=struct('Dataset',string(names{d}),'R',cfg.errorRValues(a), ...
            'MeanAOMAE',mean(aomae(:,a)),'StateSD',std(aomae(:,a),0), ...
            'MeanSeedSD',mean(seedSD(:,a)),'MeanFinalOMAE',mean(final(:,a)), ...
            'MeanSeconds',mean(seconds(:,a)));
        if isempty(rows), rows=row; else, rows(end+1)=row; end %#ok<AGROW>
    end
end
S=struct2table(rows);
end

function simulationTests(cfg)
seed=1234; U=nestedTrajectoryUniforms(seed,30,5,10);
for R=cfg.RValues
    V=nestedTrajectoryUniforms(seed,R,5,10); assert(isequal(V,U(1:R,:)));
end
stream=RandStream('mt19937ar','Seed',seed);
assert(isequal(U(1:10,1:3),rand(stream,10,3)));
bank=struct('DemandIDs',[1,2,3;1,3,4],'Importance',[1,2,3;2,0,2]);
K=[1 .2 .1 0;.2 1 .3 .2;.1 .3 1 .5;0 .2 .5 1]; ids=(1:4)';
w=bankWeights(bank,2,3,ids);
query=[1;3]; compressed=max(K(query,:),[],1)*w;
flat=0;
for t=1:2
    weights=normalizedWeights(bank.Importance(t,:)');
    flat=flat+max(K(query,bank.DemandIDs(t,:)),[],1)*weights/2;
end
assert(abs(flat-compressed)<1e-12);
[q,n]=selectBankBatch(bank,1,3,ids,K); assert(n==3 && isequal(sort(q),[1;2;3]));
[q,~]=selectBankBatch(bank,2,3,ids,K); assert(numel(unique(q))==3);
stream=RandStream('mt19937ar','Seed',9);
y=[ones(10,1);2*ones(10,1);3*ones(10,1)];
seq=nestedPoolOrder((1:30)',y,stream); assert(numel(unique(seq))==30);
assert(numel(unique(y(seq(1:6))))==3);
assert(pairJaccard({[1;2;3],[2;3;4]})==.5);
fprintf('Simulation self-tests passed: nested prefixes, common objective, union domain, pools.\n');
end

function put(fid,s)
fprintf(fid,'%s\n',s);
end


function v=pairJaccard(sets)
v=0; count=0;
for a=1:numel(sets)-1
    for b=a+1:numel(sets)
        A=unique(sets{a}); B=unique(sets{b}); den=numel(union(A,B));
        assert(den>0); v=v+numel(intersect(A,B))/den; count=count+1;
    end
end
assert(count>0); v=v/count;
end


function U=nestedTrajectoryUniforms(seed,R,k,blockSize)
stream=RandStream('mt19937ar','Seed',seed); U=zeros(R,k);
for first=1:blockSize:R
    block=rand(stream,blockSize,k); last=min(R,first+blockSize-1);
    U(first:last,:)=block(1:last-first+1,:);
end
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
% Same development-code tie convention: retention, deferment, querying.
[~,action]=min(risks,[],2);
alpha=((1-cfg.xi)*mac)./((1-cfg.xi)*mac+cfg.xi*(1-mac));
state=struct('IDs',active,'Rows',rows,'Score',score,'Alpha',alpha, ...
    'Action',action,'RAD',base.RAD(rows),'Base',base);
end



function progressReport(message,total)
persistent started completed lastPercent stage
if isempty(message)
    started=tic; completed=0; lastPercent=-1;
    chainProgress('reset');
    if isempty(stage), stage='Fixed-state simulation'; end
    fprintf('%s\n',stage);
    fprintf('Progress: 0%% | Elapsed: 00:00:00 | Remaining: --:--:--\n');
    return;
end
if isstruct(message) && isfield(message,'Kind') && strcmp(message.Kind,'stage')
    stage=message.Name; return;
end
chain=chainProgress('update',message);
isRound=isstruct(message) && isfield(message,'Kind') && strcmp(message.Kind,'round');
if isRound, completed=completed+message.Count; end
if ~isRound && ~chain.changed, return; end
percent=min(100,floor(100*completed/max(total,1)));
if percent==lastPercent && ~chain.changed, return; end
elapsed=toc(started);
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
    'MS',support,'TP',[],'RAD',rad, ...
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



function base=ensureEvidence(base,kind,context,cfg)
if strcmp(kind,'none'), return; end
assert(strcmp(kind,'TP'),'This benchmark uses only TP local evidence.');
if ~isempty(base.TP), return; end
nPool=numel(context.Universe);
activeRows=find(~ismember(context.Universe,base.ReferenceIDs));
% Coverage is model-independent: only candidate-to-reference TP distances
% are required. Avoid a full candidate-to-candidate TP matrix.
D=thresholdDistance(base.U,base.U(nPool+1:end,:),base.Models,base.Region,cfg);
base.TP=integratedEvidence(base.MS,D,base.Labels,cfg,activeRows);
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
% Retry the SAME LP; never relax tolerances or accept iteration-limited points.
opts=optimoptions('linprog','Display','none', ...
    'ConstraintTolerance',cfg.solverTolerance,'OptimalityTolerance',cfg.solverTolerance);
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
% Preserve caller RNG even if a recovery solver internally uses randomness.
originalRng=rng; rngCleanup=onCleanup(@()rng(originalRng)); %#ok<NASGU>
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



function ok=lpSolutionAccepted(x,fval,flag,res,cfg)
ok=flag>0 && ~isempty(x) && all(isfinite(x)) && ...
    isscalar(fval) && isfinite(fval) && isfinite(res) && res<=cfg.feasibilityTolerance;
end



function report=lpReport(flag,output,res,algorithm,attempts)
report=struct('ExitFlag',flag,'Message',output.message, ...
    'MaxPrimalResidual',res,'Algorithm',algorithm, ...
    'Retried',numel(attempts)>1,'Attempts',{attempts});
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
if ~converged && cfg.verbose
    fprintf('Geometry center solve stopped early; retaining the verified interior point.\n');
end
end



function [chain,audit]=sampleChain(R,start,cfg)
% Retains the supplied full-chain hit-and-run transition. Geometry is hoisted,
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


