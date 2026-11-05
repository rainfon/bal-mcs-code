function RQ3_Region_Diagnostics
%% Configuration
cfg=struct();
cfg.dataFiles={'BCC.csv','CEV.csv','ERA.csv','LEV.csv','MMG.csv', ...
    'HS.csv','NJN.csv','NZ.csv','SI.csv'};
cfg.scriptDir=fileparts(mfilename('fullpath'));
if isempty(cfg.scriptDir), cfg.scriptDir=pwd; end
cfg.dataDir=fullfile(cfg.scriptDir,'Datasets');
cfg.outputDir=cfg.scriptDir;
cfg.texName=[mfilename,'.tex'];
cfg.figName=[mfilename,'.fig'];
cfg.pdfName='Figure3.pdf';
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
cfg.verbose=false;
cfg.displayDetails=false;
cfg.progressEvery=1000;
cfg.queue=[];
cfg.progressQueue=[];
cfg.progressMethod='AVG-TP-TC';
cfg.logProgress=true;
cfg.logIntervalSeconds=20;

%% Input Validation
assert(exist('linprog','file')==2,'Optimization Toolbox with linprog is required.');
assert(cfg.numRepeats>=2 && cfg.poolCount>=cfg.rounds*cfg.batchSize);
assert(cfg.xi>0 && cfg.xi<0.5 && cfg.etaSigma>0 && cfg.epsilonPhi>0);
assert(mod(cfg.Ns,cfg.nChains)==0 && mod(cfg.chainDraws,cfg.Ns/cfg.nChains)==0);
oldPath=path;
pathGuard=onCleanup(@()path(oldPath));
addpath(cfg.scriptDir);
addpath(cfg.dataDir);
cfg.useMexSampler=exist('har_chord_mex','file')==3;
oldRng=rng;
rngGuard=onCleanup(@()rng(oldRng));
inputs=cell(numel(cfg.dataFiles),1);
for d=1:numel(inputs)
    dataPath=fullfile(cfg.dataDir,cfg.dataFiles{d});
    assert(exist(dataPath,'file')==2,'Missing CSV: %s',dataPath);
    [X,y,preprocessing]=readData(dataPath);
    [~,name]=fileparts(dataPath);
    [ref,poolIDs,test]=splitData(y,cfg.refPerClass,cfg.poolCount);
    inputs{d}=struct('X',X,'y',y,'Dataset',name,'Preprocessing',preprocessing);
    fprintf('%s: N=%d, classes=%d, reference=%d, initial pool=%d, fixed test=%d.\n', ...
        name,numel(y),max(y),numel(ref),numel(poolIDs),numel(test));
end

%% Execution
progressReport(struct('Kind','init','TotalJobs',numel(inputs)*cfg.numRepeats, ...
    'TotalStates',numel(inputs)*cfg.numRepeats*cfg.rounds, ...
    'DatasetNames',{cellfun(@(v)v.Dataset,inputs,'UniformOutput',false)}, ...
    'Interval',cfg.logIntervalSeconds));
if cfg.useParallel
    assert(license('test','Distrib_Computing_Toolbox'),'Parallel Computing Toolbox is required.');
    pool=gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers~=cfg.workers
        delete(pool);
        pool=[];
    end
    if isempty(pool)
        cluster=parcluster('local');
        cluster.NumWorkers=max(cluster.NumWorkers,cfg.workers);
        pool=parpool(cluster,cfg.workers);
    end
    if cfg.useMexSampler
        addAttachedFiles(pool,{which('har_chord_mex')});
    end
    queue=parallel.pool.DataQueue;
    afterEach(queue,@progressReport);
    cfg.progressQueue=queue;
end
nD=numel(inputs);
nJ=nD*cfg.numRepeats;
jobs=cell(nJ,1);
if cfg.useParallel
    parfor j=1:nJ
        d=floor((j-1)/cfg.numRepeats)+1;
        r=mod(j-1,cfg.numRepeats)+1;
        jobs{j}=runRegionRepeat(inputs{d},d,r,cfg);
    end
else
    for j=1:nJ
        d=floor((j-1)/cfg.numRepeats)+1;
        r=mod(j-1,cfg.numRepeats)+1;
        jobs{j}=runRegionRepeat(inputs{d},d,r,cfg);
    end
end

%% Table and Figure Output
drawnow;
[muCount,muError,muEnrichment]=writeRegionTex( ...
    fullfile(cfg.outputDir,cfg.texName),jobs,inputs,cfg);
writeRegionFigure(muCount,muError,muEnrichment,inputs,cfg);
end

%% Region Diagnostics
function job=runRegionRepeat(input,d,r,cfg)
jobClock=tic;
emitProgress(cfg,struct('Kind','start','DatasetID',d,'Repeat',r));
rng(keySeed(cfg.splitSeed,d,r,'split'),'twister');
[ref,pool,test]=splitData(input.y,cfg.refPerClass,cfg.poolCount);
ref=sort(ref); pool=sort(pool); test=sort(test);
X=input.X; y=input.y;
context=struct('DatasetID',d,'Repeat',r,'X',X,'P',basis(X,cfg.L), ...
    'q',max(y),'Universe',pool,'nCriteria',size(X,2));
Dcov=sqrt(max(0,sum((reshape(X(pool,:),numel(pool),1,[])- ...
    reshape(X(pool,:),1,numel(pool),[])).^2,3)/size(X,2)));
positive=Dcov(triu(true(numel(pool)),1)&Dcov>0);
if isempty(positive), sigmaCov=eps; else, sigmaCov=median(positive); end
context.Kcov=exp(-(Dcov/sigmaCov).^2);
context.SigmaCov=sigmaCov;
context.ModelCache=containers.Map('KeyType','char','ValueType','any');
context.ModelCache('__meta__')=struct('Order',{{}},'Bytes',0,'Hits',0);
cache=containers.Map('KeyType','char','ValueType','any');
order={}; reference=ref; labels=y(ref); active=pool;
job.Count=zeros(cfg.rounds,3);
job.ErrorRate=nan(cfg.rounds,3);
job.Enrichment=nan(cfg.rounds,3);
job.PoolError=nan(cfg.rounds,1);
[base,order]=getState(reference,labels,context,cfg,cache,order,true,'none');
for t=1:cfg.rounds
    base=ensureEvidence(base,'TP',context,cfg);
    cache(base.StateKey)=base;
    state=mainState(base,active,context,cfg);
    job=recordRegionState(job,t,state,active,y);
    emitProgress(cfg,struct('Kind','round','DatasetID',d,'Repeat',r,'Round',t, ...
        'Count',job.Count(t,:),'Rate',job.ErrorRate(t,:), ...
        'Enrichment',job.Enrichment(t,:),'PoolError',job.PoolError(t)));
    if t==cfg.rounds, break; end
    [query,order]=mainBatch(state,reference,labels,active,context,cfg,cache,order,t);
    assert(numel(query)==min(cfg.batchSize,numel(active)) && ...
        numel(unique(query))==numel(query) && all(ismember(query,active)));
    assert(~any(ismember(query,test)),'Test sample entered query batch.');
    reference=[reference;query]; labels=[labels;y(query)];
    [reference,ix]=sort(reference); labels=labels(ix);
    active=setdiff(active,query,'stable');
    [base,order]=getState(reference,labels,context,cfg,cache,order,true,'none');
end
emitProgress(cfg,struct('Kind','repeat','DatasetID',d,'Repeat',r,'Seconds',toc(jobClock)));
end

%% Regional Statistics
function job=recordRegionState(job,t,state,active,y)
wrong=state.Hat~=y(active);
poolError=mean(wrong);
job.PoolError(t)=poolError;
for a=1:3
    member=state.Action==a;
    job.Count(t,a)=sum(member);
    if any(member)
        job.ErrorRate(t,a)=mean(wrong(member));
        if poolError>0
            job.Enrichment(t,a)=job.ErrorRate(t,a)/poolError;
        end
    end
end
assert(sum(job.Count(t,:))==numel(active));
end

%% LaTeX Table
function [muCount,muError,muEnrichment]=writeRegionTex(outputPath,jobs,inputs,cfg)
nD=numel(inputs);
muCount=nan(nD,3); muError=muCount; muEnrichment=muCount;
validRate=zeros(nD,3); validEnrichment=validRate;
for d=1:nD
    subset=jobs((d-1)*cfg.numRepeats+(1:cfg.numRepeats));
    counts=cellfun(@(j)j.Count,subset,'UniformOutput',false);
    rates=cellfun(@(j)j.ErrorRate,subset,'UniformOutput',false);
    enrichments=cellfun(@(j)j.Enrichment,subset,'UniformOutput',false);
    C=vertcat(counts{:}); E=vertcat(rates{:}); G=vertcat(enrichments{:});
    muCount(d,:)=mean(C,1);
    for a=1:3
        ok=isfinite(E(:,a)); validRate(d,a)=sum(ok);
        if any(ok), muError(d,a)=mean(E(ok,a)); end
        ok=isfinite(G(:,a)); validEnrichment(d,a)=sum(ok);
        if any(ok), muEnrichment(d,a)=mean(G(ok,a)); end
    end
end
fid=fopen(outputPath,'w');
assert(fid>=0,'Cannot write %s.',outputPath);
fileGuard=onCleanup(@()fclose(fid));
fprintf(fid,'%% Requires: booktabs.\n');
fprintf(fid,'%% AVG-TP-TC; pre-query states in rounds 1--%d; %d repeats per dataset.\n',cfg.rounds,cfg.numRepeats);
fprintf(fid,'%% Count: mean across all repeat-round states, including empty regions.\n');
fprintf(fid,'%% Error rate: arithmetic mean of within-region rates over nonempty states.\n');
fprintf(fid,'%% Enrichment: arithmetic mean of state-wise regional/pool error-rate ratios;\n');
fprintf(fid,'%% states with an empty region or zero pool error are excluded from enrichment.\n');
fprintf(fid,'%% L=%d; Ns=%d; R=%d; rounds=%d; batch=%d; pool=%d; initial references/class=%d.\n', ...
    cfg.L,cfg.Ns,cfg.numTrajectories,cfg.rounds,cfg.batchSize,cfg.poolCount,cfg.refPerClass);
fprintf(fid,'%% xi=%.6g; etaSigma=%.6g; epsilonAssignment=%.6g; epsilonPhi=%.6g.\n', ...
    cfg.xi,cfg.etaSigma,cfg.epsilonAssignment,cfg.epsilonPhi);
fprintf(fid,'%% chains=%d; burn-in/chain=%d; production/chain=%d; retained/chain=%d; thinning=%d.\n', ...
    cfg.nChains,cfg.burnIn,cfg.chainDraws,cfg.Ns/cfg.nChains, ...
    cfg.chainDraws/(cfg.Ns/cfg.nChains));
fprintf(fid,'%% splitSeed=%d; chainSeed=%d; feedbackSeed=%d; MEX=%d.\n', ...
    cfg.splitSeed,cfg.chainSeed,cfg.feedbackSeed,cfg.useMexSampler);
regions={'POS','BND','NEG'};
for d=1:nD
    for a=1:3
        fprintf(fid,'%% %s %s: valid error-rate states=%d/%d; valid enrichment states=%d/%d.\n', ...
            inputs{d}.Dataset,regions{a},validRate(d,a),cfg.numRepeats*cfg.rounds, ...
            validEnrichment(d,a),cfg.numRepeats*cfg.rounds);
    end
end
fprintf(fid,'\\begin{table}[htbp]\n\\centering\n');
fprintf(fid,'\\caption{Region-wise diagnostics of AVG--TP--TC before each query round.}\n');
fprintf(fid,'\\label{tab:region_diagnostics}\n');
fprintf(fid,'\\begin{tabular}{llrrr}\n\\toprule\n');
fprintf(fid,'Dataset & Region & Average size & Error rate & Error enrichment \\\\\n');
fprintf(fid,'\\midrule\n');
for d=1:nD
    for a=1:3
        if a==1, dataset=texEscape(inputs{d}.Dataset); else, dataset=''; end
        fprintf(fid,'%s & %s & %.2f & %s & %s \\\\\n',dataset,regions{a},muCount(d,a), ...
            tableNumber(muError(d,a)),tableNumber(muEnrichment(d,a)));
    end
    if d<nD, fprintf(fid,'\\addlinespace\n'); end
end
fprintf(fid,'\\bottomrule\n\\end{tabular}\n');
fprintf(fid,'\\par\\smallskip\n\\begin{minipage}{0.96\\linewidth}\n\\footnotesize\n');
fprintf(fid,['Notes: All diagnostics use real information states immediately before querying; ', ...
    'hypothetical trajectory states are excluded. Average size includes zero-size regions. ', ...
    'Error rates are averaged over nonempty regions. Error enrichment is the mean ', ...
    'of the state-wise regional error rate divided by the active-pool error rate; ', ...
    'states with zero pool error are excluded. Undefined summaries are shown as --. ', ...
    'Benchmark labels are used only for diagnostics and elicited feedback. ', ...
    'Results summarize %d paired repetitions and %d pre-query rounds per dataset.\n'], ...
    cfg.numRepeats,cfg.rounds);
fprintf(fid,'\\end{minipage}\n\\end{table}\n');
end

%% Figure from the same summaries written to the LaTeX table
function writeRegionFigure(muCount,muError,muEnrichment,inputs,cfg)
datasets=cellfun(@(v)v.Dataset,inputs,'UniformOutput',false);
regions={'POS','BND','NEG'};
values={muCount,muError,muEnrichment};
titles={'(a) Average region size','(b) Regional error rate', ...
    '(c) Error enrichment'};
ylimits={[0,42],[0,0.62],[0,5.1]};
yticks={0:10:40,0:0.1:0.6,0:1:5};
colors=[132 192 224;28 148 180;252 172 4]/255;
lineColor=[0.25 0.25 0.25];
nD=numel(datasets);
assert(all(cellfun(@(v)isequal(size(v),[nD,3]),values)));

figureWidth=67.7333333333333;
figureHeight=29.5275;
fig=figure('Visible','on','Color','w','Units','centimeters', ...
    'Position',[2,2,figureWidth,figureHeight],'PaperUnits','centimeters', ...
    'PaperPosition',[0,0,figureWidth,figureHeight], ...
    'PaperSize',[figureWidth,figureHeight]);
figureGuard=onCleanup(@()close(fig));
layout=tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
handles=gobjects(1,3);
for panel=1:3
    ax=nexttile(layout,panel);
    hold(ax,'on');
    bars=bar(ax,1:nD,values{panel},'grouped','BarWidth',0.82);
    for region=1:3
        bars(region).FaceColor=colors(region,:);
        bars(region).EdgeColor='none';
    end
    if panel==1, handles=bars; end
    if panel==3
        yline(ax,1,'--','Color',lineColor, ...
            'LineWidth',0.9,'HandleVisibility','off');
    end
    ax.XLim=[0.5,nD+0.5];
    ax.YLim=ylimits{panel};
    ax.XTick=1:nD;
    ax.XTickLabel=datasets;
    ax.YTick=yticks{panel};
    ax.XTickLabelRotation=55;
    ax.FontName='Arial';
    ax.FontSize=30;
    ax.LineWidth=2;
    ax.Box='off';
    ax.XGrid='off';
    ax.YGrid='on';
    ax.GridColor=[0.84 0.84 0.84];
    ax.GridAlpha=0.7;
    ax.Layer='bottom';
    title(ax,titles{panel},'FontWeight','normal','FontSize',30);
end
legendHandle=legend(handles,regions,'Orientation','horizontal', ...
    'Box','off','FontSize',30);
legendHandle.Layout.Tile='north';
set(findall(fig,'Type','axes'),'Box','on','LineWidth',2);
savefig(fig,fullfile(cfg.outputDir,cfg.figName));
exportgraphics(fig,fullfile(figureDirs(cfg.outputDir),cfg.pdfName),'ContentType','vector');
end

%% Number Formatting
function value=tableNumber(x)
if isfinite(x), value=sprintf('%.4f',x); else, value='--'; end
end

%% Label Formatting
function out=texEscape(in)
out=char(in);
out=strrep(out,'_','\_');
out=strrep(out,'&','\&');
out=strrep(out,'%','\%');
out=strrep(out,'#','\#');
end

%% mainState
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
            'Repeat',context.Repeat,'Chain',ch,'RealState',isRealState,'ReferenceCount',numel(ref));
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
q=size(p,2); distance=abs(hat(:)-(1:q));
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

%% readData
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

%% splitData
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

%% Progress Events
function emitProgress(cfg,packet)
if ~cfg.logProgress, return; end
if isempty(cfg.progressQueue)
    progressReport(packet);
else
    send(cfg.progressQueue,packet);
end
end

%% Progress Reporting
function progressReport(packet)
persistent clock totalStates completedStates lastPercent
if strcmp(packet.Kind,'init')
    clock=tic; totalStates=packet.TotalStates; completedStates=0; lastPercent=-1;
    chainProgress('reset');
    fprintf('Progress: 0%% | Elapsed: 00:00:00 | Remaining: --:--:--\n');
    return;
end
chain=chainProgress('update',packet);
if strcmp(packet.Kind,'round'), completedStates=completedStates+1; end
if ~strcmp(packet.Kind,'round') && ~chain.changed, return; end
percent=min(100,floor(100*completedStates/max(totalStates,1)));
if percent==lastPercent && ~chain.changed, return; end
elapsed=toc(clock);
if completedStates>0
    remaining=formatDuration(elapsed*(totalStates-completedStates)/completedStates);
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
