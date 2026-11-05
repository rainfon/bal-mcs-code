function RQ4_Diagnostics_Nine
% Table 12 protocol on the nine evaluation datasets.
% Same screen as RQ4_Ns_Sensitivity: four chains, burn-in 100,
% 200000 production transitions, every 400th state retained (Ns=2000).
% One reference set per dataset: three references per class, pool of 50.
% Split seed is 20261009 plus the dataset index in the list below.
% Requires har_chord_mex, Optimization Toolbox, and Parallel Computing Toolbox.
clc;
cfg=struct();
cfg.dataFiles={'BCC.csv','CEV.csv','ERA.csv','LEV.csv','MMG.csv', ...
    'HS.csv','NJN.csv','NZ.csv','SI.csv'};
scriptDir=fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir=pwd; end
addpath(scriptDir);
assert(exist('har_chord_mex','file')==3, ...
    'Place the compiled har_chord_mex beside this program or on the MATLAB path.');
cfg.outputDir=scriptDir;
cfg.dataDir=fullfile(scriptDir,'Datasets');
if ~isfolder(cfg.dataDir), cfg.dataDir=scriptDir; end
cfg.workers=16; cfg.refPerClass=3; cfg.poolCount=50; cfg.L=3;
cfg.splitSeed=20261009; cfg.chainSeed=271828;
cfg.epsilonAssignment=1e-6; cfg.epsilonPhi=1e-5;
cfg.nChains=4; cfg.Ns=2000;
cfg.burnIn=100;
cfg.checkpointDraws=200000;
cfg.chainDraws=cfg.checkpointDraws(end);
cfg.usePreconditioning=true; cfg.preconditioningMethod='hessian';
cfg.pilotRounds=2; cfg.pilotNs=4000; cfg.pilotBurnIn=1000;
cfg.covarianceFloor=1e-5;
cfg.directionStrategy='coordinate_mixture'; cfg.sphereProbability=0.2;
cfg.solverTolerance=1e-9; cfg.feasibilityTolerance=1e-9;
cfg.minimumRadius=1e-12; cfg.lineIterations=60;
cfg.barrierMaxIterations=80; cfg.barrierNewtonTolerance=1e-9;
cfg.barrierArmijo=0.01; cfg.barrierStepFraction=0.99;
cfg.rhatLimit=1.01; cfg.essMinimum=400; cfg.supportMCSETarget=0.02;
cfg.queue=[]; cfg.verbose=false; cfg.displayDetails=false;
cfg.computeCheckpoints=false;
cfg.mode='diagnose'; cfg.phase='full'; cfg.runSelfTest=true;
cfg.progressEvery=10000;
validateConfig(cfg);
assert(all(diff(cfg.checkpointDraws)>0)&&all(mod(cfg.checkpointDraws,500)==0), ...
    'Checkpoints must increase and be divisible by 500.');
assert(exist('linprog','file')==2,'Optimization Toolbox required.');
assert(license('test','Distrib_Computing_Toolbox'),'Parallel Computing Toolbox required.');
oldRng=rng; rngCleanup=onCleanup(@()rng(oldRng)); %#ok<NASGU>
if cfg.runSelfTest
    tests=selfTests(cfg); assert(all(tests.Pass),'HAR:SelfTest','Sampler self-test failed.');
end
p=gcp('nocreate');
if ~isempty(p)&&p.NumWorkers~=cfg.workers, delete(p); p=[]; end
if isempty(p)
    cluster=parcluster('local'); cluster.NumWorkers=max(cluster.NumWorkers,cfg.workers);
    p=parpool(cluster,cfg.workers);
end
nD=numel(cfg.dataFiles);
progressQueue=parallel.pool.DataQueue;
progressTotal=nD*cfg.nChains*(cfg.burnIn+cfg.chainDraws);
afterEach(progressQueue,@(increment)reportSamplingProgress(increment,progressTotal));
inputs=cell(nD,1); seedNames={'BCC','CEV','ERA','LEV','MMG','HS','NJN','NZ','SI'};
for di=1:nD
    file=fullfile(cfg.dataDir,cfg.dataFiles{di}); assert(exist(file,'file')==2,'Missing CSV: %s',file);
    [X,y,preprocessing]=readData(file); [~,dataset]=fileparts(file);
    seedID=find(strcmpi(dataset,seedNames),1); rng(cfg.splitSeed+seedID,'twister');
    [ref,pool,test]=splitData(y,cfg.refPerClass,cfg.poolCount);
    fprintf('%s: N=%d, classes=%d, reference=%d, initial pool=%d, fixed test=%d.\n', ...
        dataset,numel(y),max(y),numel(ref),numel(pool),numel(test));
    inputs{di}=struct('Dataset',dataset,'X',X,'y',y,'ReferenceIDs',ref, ...
        'PoolIDs',pool,'TestIDs',test,'Preprocessing',preprocessing,'SeedID',seedID);
end
reportSamplingProgress(0,progressTotal);
cases=cell(nD,1);
parfor di=1:nD
    one=inputs{di}; startClock=tic;
    [R,lpInfo]=makeRegion(one.X(one.ReferenceIDs,:),one.y(one.ReferenceIDs),max(one.y),cfg);
    seeds=cfg.chainSeed+10000*one.SeedID+100+(1:cfg.nChains);
    [starts,startInfo]=makeStarts(R,seeds,cfg,string(one.Dataset));
    preClock=tic; [R,preconditioning]=preparePreconditioner(R,starts,seeds,cfg);
    one.Region=R; one.LPInfo=lpInfo; one.Starts=starts; one.StartInfo=startInfo;
    one.Seeds=seeds; one.Preconditioning=preconditioning;
    one.PreconditioningSeconds=toc(preClock); one.PreparationSeconds=toc(startClock);
    cases{di}=one;
end
chains=cell(nD,cfg.nChains); states=cell(nD,cfg.nChains);
starts=cell(nD,cfg.nChains); audits=cell(nD,cfg.nChains);
for di=1:nD
    for c=1:cfg.nChains, starts{di,c}=cases{di}.Starts(:,c); end
end
active=true(nD,1); previousPass=false(nD,1); confirmed=false(nD,1);
draws=zeros(nD,1); samplingSeconds=zeros(nD,1); diagnosticSeconds=zeros(nD,1);
checkpoints=cell(nD,1); jobs=cell(nD,1); failures=cell(nD,1);
for stage=1:numel(cfg.checkpointDraws)
    ids=find(active); if isempty(ids), break; end
    target=cfg.checkpointDraws(stage); nTasks=numel(ids)*cfg.nChains;
    blocks=cell(nTasks,1); blockStates=blocks; blockAudits=blocks; errors=blocks;
    parfor ti=1:nTasks
        di=ids(ceil(ti/cfg.nChains)); c=mod(ti-1,cfg.nChains)+1;
        blockCfg=cfg; blockCfg.queue=progressQueue; blockCfg.chainDraws=target-draws(di);
        if draws(di)>0, blockCfg.burnIn=0; end
        try
            if draws(di)==0, rng(cases{di}.Seeds(c),'twister'); else, rng(states{di,c}); end
            [block,audit]=sampleChain(cases{di}.Region,starts{di,c},blockCfg);
            blocks{ti}=block; blockStates{ti}=rng; blockAudits{ti}=audit;
        catch ME
            errors{ti}=struct('Dataset',string(cases{di}.Dataset),'ErrorID',string(ME.identifier), ...
                'ErrorMessage',string(ME.message));
        end
    end
    for ii=1:numel(ids)
        di=ids(ii); taskIDs=(ii-1)*cfg.nChains+(1:cfg.nChains);
        bad=find(~cellfun(@isempty,errors(taskIDs)),1);
        if ~isempty(bad)
            failures{di}=errors{taskIDs(bad)}; active(di)=false; continue;
        end
        blockTimes=zeros(cfg.nChains,1);
        for c=1:cfg.nChains
            ti=taskIDs(c); chains{di,c}=[chains{di,c};blocks{ti}];
            states{di,c}=blockStates{ti}; starts{di,c}=blocks{ti}(end,:)';
            blockTimes(c)=blockAudits{ti}.Seconds;
            if draws(di)==0, audits{di,c}=blockAudits{ti};
            else, audits{di,c}=mergeLightAudit(audits{di,c},blockAudits{ti}); end
        end
        samplingSeconds(di)=samplingSeconds(di)+max(blockTimes);
        draws(di)=target;
    end
    diagIDs=ids(active(ids)); evaluated=cell(numel(diagIDs),1);
    parfor ii=1:numel(diagIDs)
        di=diagIDs(ii); t=tic;
        try
            chain=cat(3,chains{di,:}); evalJob=evaluateLight(chain,cases{di},audits(di,:),cfg);
            evalJob.DiagnosticSeconds=toc(t); evaluated{ii}=evalJob;
        catch ME
            evaluated{ii}=struct('Success',false,'Dataset',string(cases{di}.Dataset), ...
                'ErrorID',string(ME.identifier),'ErrorMessage',string(ME.message));
        end
    end
    for ii=1:numel(diagIDs)
        di=diagIDs(ii); job=evaluated{ii};
        if ~job.Success, failures{di}=job; active(di)=false; continue; end
        diagnosticSeconds(di)=diagnosticSeconds(di)+job.DiagnosticSeconds;
        row=job.Row; checkpoints{di}=appendStruct(checkpoints{di},row);
        confirmed(di)=row.ScreenPass&&previousPass(di); previousPass(di)=row.ScreenPass;
        
        row.PreparationSeconds=cases{di}.PreparationSeconds;
        row.PreconditioningSeconds=cases{di}.PreconditioningSeconds;
        row.ChainParallelSeconds=samplingSeconds(di); row.DiagnosticSeconds=diagnosticSeconds(di);
        row.EstimatedDatasetSeconds=row.PreparationSeconds+samplingSeconds(di)+diagnosticSeconds(di);
        row.ESSPerEstimatedSecond=row.MinBulkESS/row.EstimatedDatasetSeconds;
        job.Row=row; jobs{di}=job;
        % Single final checkpoint; no per-dataset early stopping.
    end
    clear blocks blockStates blockAudits evaluated
end
summaryRows=struct([]); checkpointRows=struct([]); failureRows=struct([]);
for di=1:nD
    if ~isempty(jobs{di}), summaryRows=appendStruct(summaryRows,jobs{di}.Row); end
    for k=1:numel(checkpoints{di}), checkpointRows=appendStruct(checkpointRows,checkpoints{di}(k)); end
    if ~isempty(failures{di})
        f=failures{di}; failureRows=appendStruct(failureRows, ...
            struct('Dataset',f.Dataset,'ErrorID',f.ErrorID,'ErrorMessage',f.ErrorMessage));
    end
end
summary=table(); checkpointTable=table(); failureTable=table();
if ~isempty(summaryRows), summary=struct2table(summaryRows); end
if ~isempty(checkpointRows), checkpointTable=struct2table(checkpointRows); end
if ~isempty(failureRows), failureTable=struct2table(failureRows); end
% Keep diagnostics in memory for the table; do not write a MAT file.
for di=1:nD, cases{di}.X=[]; cases{di}.y=[]; end
results=struct('Configuration',cfg,'Summary',summary,'Checkpoints',checkpointTable, ...
    'Failures',failureTable,'Cases',{cases},'Jobs',{jobs});
if height(results.Summary)~=nD
    if ~isempty(results.Failures), disp(results.Failures); end
    error('RQ4:DiagnosticFailure','Diagnostics did not finish successfully for every dataset.');
end
texPath=fullfile(cfg.outputDir,'RQ4_Diagnostics_Nine.tex');
writeLatexDiagnostics(results.Summary,texPath);
end
function job=evaluateLight(chain,one,auditCells,cfg)
R=one.Region; N=size(chain,1); P=basis(one.X(one.PoolIDs,:),cfg.L);
parameterCfg=cfg; parameterCfg.parameterOnly=true;
[D,~,~,~]=analyseChains(chain,R,P,one.PoolIDs,parameterCfg);
thin=N/(cfg.Ns/cfg.nChains); kept=chain(thin:thin:N,:,:);
[~,S,~,~]=analyseChains(kept,R,P,one.PoolIDs,cfg);
a=D.Status=="ok"; b=S.Status=="ok";
failures=0; for c=1:numel(auditCells), failures=failures+auditCells{c}.NumericalFailures; end
pass=any(a)&&any(b)&&all(isfinite(D.Rhat(a)))&&all(D.Rhat(a)<cfg.rhatLimit) ...
    &&all(isfinite(D.BulkESS(a)))&&all(D.BulkESS(a)>=cfg.essMinimum) ...
    &&all(isfinite(D.TailESS(a)))&&all(D.TailESS(a)>=cfg.essMinimum) ...
    &&all(isfinite(S.Rhat(b)))&&all(S.Rhat(b)<cfg.rhatLimit) ...
    &&all(isfinite(S.MCSE(b)))&&all(S.MCSE(b)<=cfg.supportMCSETarget)&&failures==0;
models=reshape(permute(kept,[2,1,3]),R.nVar,[]); assert(size(models,2)==2000);
avg=mean(models,2); assertFeasible(avg,R,cfg);
U=basis(one.X(one.TestIDs,:),cfg.L)*avg(1:R.p); pred=ones(size(U));
for h=1:R.q-1, pred=pred+(U>avg(R.p+h)); end
omae=mean(abs(pred-one.y(one.TestIDs)))/(R.q-1);
row=struct('Dataset',string(one.Dataset),'EpsilonPhi',cfg.epsilonPhi,'Ns',cfg.Ns, ...
    'DrawsPerChain',N,'Thin',thin,'MaxRhat',finiteMax(D.Rhat), ...
    'MinBulkESS',finiteMin(D.BulkESS(a)),'MinTailESS',finiteMin(D.TailESS(a)), ...
    'MaxSupportRhat',finiteMax(S.Rhat(b)),'MinSupportESS',finiteMin(S.MeanESS(b)), ...
    'MaxSupportMCSE',finiteMax(S.MCSE(b)),'ConstantParameterCells',sum(~a), ...
    'ConstantSupportCells',sum(~b),'NumericalFailures',failures,'ScreenPass',pass,'OMAE',omae);
job=struct('Success',true,'Row',row,'ParameterDiagnostics',D,'SupportDiagnostics',S, ...
    'Models',models,'TestAVGAssignment',pred,'Audit',struct2table(vertcat(auditCells{:})));
end
function writeLatexDiagnostics(T,path)
fid = fopen(path,'w');
assert(fid>0,'HAR:TeX','Cannot write LaTeX table: %s',path);
fileCleanup = onCleanup(@()fclose(fid));
eol = [char(92),char(92)];
fprintf(fid,'%s\n','\begin{table}[H]');
fprintf(fid,'%s\n','\centering');
fprintf(fid,'%s\n','\caption{Mixing of the hit-and-run chains on nine evaluation datasets.}');
fprintf(fid,'%s\n','\label{tab:har_diagnostics_nine}');
fprintf(fid,'%s\n','\fontsize{7.8pt}{12pt}\selectfont');
fprintf(fid,'%s\n','\renewcommand{\arraystretch}{1.12}');
fprintf(fid,'%s\n','\begin{tabular}{lrrrrrrr}');
fprintf(fid,'%s\n','\toprule');
fprintf(fid,'%s\n',['& \multicolumn{3}{c}{Full chain} & \multicolumn{4}{c}{Class support}',eol]);
fprintf(fid,'%s\n','\cmidrule(lr){2-4}\cmidrule(lr){5-8}');
fprintf(fid,'%s\n',['Dataset & Max $\widehat R$ & Min bulk ESS & Min tail ESS & Max $\widehat R$ & Min ESS & Max MCSE & Constant cells',eol]);
fprintf(fid,'%s\n','\midrule');
for i=1:height(T)
    fprintf(fid,'%s\n',[sprintf('%s & %.4f & %.2f & %.2f & %.4f & %.2f & %.5f & %.0f', ...
        char(T.Dataset(i)),T.MaxRhat(i),T.MinBulkESS(i),T.MinTailESS(i), ...
        T.MaxSupportRhat(i),T.MinSupportESS(i),T.MaxSupportMCSE(i), ...
        T.ConstantSupportCells(i)),eol]);
end
fprintf(fid,'%s\n','\bottomrule');
fprintf(fid,'%s\n','\end{tabular}');
fprintf(fid,'%s\n','\end{table}');
end
function reportSamplingProgress(increment,total)
persistent done clock lastPercent
if increment==0
    done=0; clock=tic; lastPercent=-1;
    fprintf('Progress: 0%% | Elapsed: 00:00:00 | Remaining: --:--:--\n');
    return;
end
done=done+increment;
percent=min(100,floor(100*done/max(total,1)));
if percent==lastPercent, return; end
elapsed=toc(clock);
if done>0
    remaining=formatDuration(elapsed*(total-done)/done);
else
    remaining='--:--:--';
end
fprintf('Progress: %d%% | Elapsed: %s | Remaining: %s\n', ...
    percent,formatDuration(elapsed),remaining);
lastPercent=percent;
end
function text=formatDuration(seconds)
seconds=max(0,round(seconds));
text=sprintf('%02d:%02d:%02d',floor(seconds/3600),floor(mod(seconds,3600)/60),mod(seconds,60));
end
function printDiagnosticTable(T)
if isempty(T), fprintf('No successful diagnostic results.\n'); return; end
fprintf('%-8s %36s %42s\n','','Full chain','Class support');
fprintf('%-8s %10s %12s %12s %10s %10s %10s %8s\n', ...
    'Dataset','MaxR','MinBulkESS','MinTailESS','MaxR','MinESS','MaxMCSE','Cells');
for i=1:height(T)
    fprintf('%-8s %10.4f %12.2f %12.2f %10.4f %10.2f %10.5f %8.0f\n', ...
        char(T.Dataset(i)),T.MaxRhat(i),T.MinBulkESS(i),T.MinTailESS(i), ...
        T.MaxSupportRhat(i),T.MinSupportESS(i),T.MaxSupportMCSE(i),T.ConstantSupportCells(i));
end
end
function a=mergeLightAudit(a,b)
for k={'Seconds','Transitions','ContinuousDraws','NumericalFailures'}
    n=k{1}; a.(n)=a.(n)+b.(n);
end
for k={'MinChord','MinStep'}, n=k{1}; a.(n)=min(a.(n),b.(n)); end
for k={'MaxChord','MaxStructuralResidual','MaxEqualityResidual','MaxLossResidual'}
    n=k{1}; a.(n)=max(a.(n),b.(n));
end
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
opts=optimoptions('linprog','Display','none', ...
    'ConstraintTolerance',cfg.solverTolerance,'OptimalityTolerance',cfg.solverTolerance);
[x,fval,flag,output]=linprog(f,sparse(A),b,sparse(Aeq),beq,lb,ub,opts);
assert(flag>0&&~isempty(x)&&all(isfinite(x)),'HAR:LP','linprog failed: %s',output.message);
res=max([0;A*x-b;abs(Aeq*x-beq);lb-x]);
if ~isempty(ub), res=max([res; x-ub]); end
assert(res<=cfg.feasibilityTolerance,'HAR:LPResidual','LP residual %g exceeds tolerance.',res);
report=struct('ExitFlag',flag,'Message',output.message,'MaxPrimalResidual',res);
end
%% 6. Sampling and geometric preconditioning
function [starts,info]=makeStarts(R,seeds,cfg,dataset)
if nargin<4, dataset="unknown"; end
starts=zeros(R.nVar,numel(seeds)); info=struct([]);
for c=1:numel(seeds)
    rng(seeds(c)+17,'twister'); objective=[R.B*randn(size(R.B,2),1);zeros(numel(R.y),1)];
    [sol,~,report]=solveLP(objective,R.lpA,R.lpRhs,R.lpEq,1,R.lpLb,[],cfg);
    starts(:,c)=0.7*R.center+0.3*sol(1:R.nVar);
    if ~startIsDistinct(starts(:,c),starts(:,1:c-1),1e-12)
        starts(:,c)=separatedInteriorStart(R,c,starts(:,1:c-1));
    end
    assertFeasible(starts(:,c),R,cfg);
    report.DistanceFromCenter=norm(starts(:,c)-R.center); info=appendStruct(info,report);
end
gap=minimumStartGap(starts);
assert(gap>1e-12,'HAR:Starts', ...
    'Distinct starts could not be constructed for %s. Radius %.3e, minimum gap %.3e.', ...
    dataset,R.radius,gap);
end
function yes=startIsDistinct(x,previous,tol)
yes=true;
for k=1:size(previous,2)
    if norm(x-previous(:,k))<=tol, yes=false; return; end
end
end
function gap=minimumStartGap(starts)
gap=inf;
for a=1:size(starts,2)
    for b=a+1:size(starts,2)
        gap=min(gap,norm(starts(:,a)-starts(:,b)));
    end
end
end
function x=separatedInteriorStart(R,index,previous)
% Random-objective vertices can coincide. The inscribed ball is certified
% feasible, so a start is placed inside that ball, away from starts already chosen.
d=size(R.B,2);
scales=[0.55,-0.40,0.25,-0.70,0.15,-0.85,0.35,-0.60];
for attempt=0:d*numel(scales)-1
    z=zeros(d,1);
    slot=mod(index-1+attempt,d)+1;
    z(slot)=scales(mod(floor(attempt/d),numel(scales))+1)*R.radius;
    x=R.center+R.B*z;
    if startIsDistinct(x,previous,1e-12), return; end
end
end
function [R,report]=preparePreconditioner(R,starts,seeds,cfg)
report=table(0,0,1,0,'VariableNames', ...
    {'Round','Seconds','RegularizedCovarianceCondition','NumericalFailures'});
if ~cfg.usePreconditioning
    R.DirectionBasis=R.B;
    return;
end
if strcmp(cfg.preconditioningMethod,'hessian')
    [R,report]=prepareHessianGeometry(R,cfg); return;
end
if strcmp(cfg.preconditioningMethod,'hybrid')
    [R,report]=prepareHybridGeometry(R,starts,seeds,cfg); return;
end
pilotCfg=cfg; pilotCfg.queue=[]; pilotCfg.chainDraws=cfg.pilotNs; pilotCfg.burnIn=cfg.pilotBurnIn;
rows=struct([]);
for round=1:cfg.pilotRounds
    t0=tic; pooled=zeros(cfg.pilotNs*cfg.nChains,size(R.B,2)); failures=0;
    pilotCells=cell(cfg.nChains,1); pilotFailures=zeros(cfg.nChains,1);
   for c=1:cfg.nChains
       rng(seeds(c)+1000000+10000*round,'twister');
       [draws,audit]=sampleChain(R,starts(:,c),pilotCfg);
       pilotCells{c}=(draws-R.center')*R.B;
       pilotFailures(c)=audit.NumericalFailures;
   end
   pooled=vertcat(pilotCells{:}); failures=sum(pilotFailures);
   assert(failures==0,'HAR:PilotFailure','Pilot had numerical failures; inspect geometry.');
    covariance=cov(pooled,1); covariance=(covariance+covariance')/2;
    [vectors,values]=eig(covariance); eigenvalues=real(diag(values));
    largest=max(eigenvalues);
    assert(isfinite(largest)&&largest>0,'HAR:PilotConstant','Pilot has no usable spread.');
    eigenvalues=max(eigenvalues,cfg.covarianceFloor*largest);
    factor=vectors*diag(sqrt(eigenvalues/largest));
    R.DirectionBasis=R.B*factor;
    condition=max(eigenvalues)/min(eigenvalues);
    row=struct('Round',round,'Seconds',toc(t0), ...
        'RegularizedCovarianceCondition',condition,'NumericalFailures',failures);
    rows=appendStruct(rows,row);
end
report=struct2table(rows);
end
function [R,report]=prepareHybridGeometry(R,starts,seeds,cfg)
[R,hessianReport]=prepareHessianGeometry(R,cfg);
R.HessianReport=hessianReport;
pilotCfg=cfg; pilotCfg.queue=[]; pilotCfg.chainDraws=cfg.pilotNs; pilotCfg.burnIn=cfg.pilotBurnIn;
rows=struct([]); warmStarts=starts;
for round=1:cfg.pilotRounds
    t0=tic; map=R.DirectionBasis; [Q,T]=qr(map,0);
    assert(rcond(T)>1e-14,'HAR:RoundSingular','Rounding map numerically singular.');
    d=size(map,2); pooled=zeros(cfg.pilotNs*cfg.nChains,d); failures=0;
    pilotCells=cell(cfg.nChains,1); pilotFailures=zeros(cfg.nChains,1);
   newStarts=zeros(size(warmStarts));
   for c=1:cfg.nChains
       rng(seeds(c)+2000000+10000*round,'twister');
       [draws,audit]=sampleChain(R,warmStarts(:,c),pilotCfg);
       pilotCells{c}=(draws-R.center')*Q/T';
       newStarts(:,c)=draws(end,:)';
       pilotFailures(c)=audit.NumericalFailures;
   end
   pooled=vertcat(pilotCells{:}); failures=sum(pilotFailures);
   warmStarts=newStarts;
   assert(failures==0,'HAR:PilotFailure','Full-region pilot had numerical failures.');
    C=cov(pooled,1); C=(C+C')/2; [V,D]=eig(C); eigenvalues=real(diag(D));
    largest=max(eigenvalues);
    assert(all(isfinite(eigenvalues))&&largest>0,'HAR:PilotConstant','Invalid pilot covariance.');
    floored=sum(eigenvalues<cfg.covarianceFloor*largest);
    adjusted=max(eigenvalues,cfg.covarianceFloor*largest);
    correction=V*diag(sqrt(adjusted/largest));
    R.DirectionBasis=map*correction;
    R.DirectionBasis=R.DirectionBasis/norm(R.DirectionBasis,'fro');
    assert(all(isfinite(R.DirectionBasis(:))),'HAR:RoundFinite','Nonfinite rounding map.');
    singularValues=svd(R.DirectionBasis);
    assert(min(singularValues)>1e-14*max(singularValues), ...
        'HAR:RoundSingular','Rounding map has lost numerical rank.');
    row=struct('Round',round,'Seconds',toc(t0), ...
        'RegularizedCovarianceCondition',max(adjusted)/min(adjusted), ...
        'NumericalFailures',failures,'FlooredEigenvalues',floored, ...
        'MapCondition',max(singularValues)/min(singularValues));
    rows=appendStruct(rows,row);
    if isfield(cfg,'pilotConditionTarget') && floored==0 && ...
            max(adjusted)/min(adjusted)<=cfg.pilotConditionTarget
        break; % Only ends pilot adaptation; does not bypass production diagnostics.
    end
    if isfield(cfg,'pilotConditionTarget') && floored==0 && ...
            max(adjusted)/min(adjusted)<=cfg.pilotConditionTarget
        break; % Only ends pilot adaptation; does not bypass production diagnostics.
    end
    if isfield(cfg,'pilotConditionTarget') && floored==0 && ...
            max(adjusted)/min(adjusted)<=cfg.pilotConditionTarget
        break; % Only ends pilot adaptation; does not bypass production diagnostics.
    end
end
report=struct2table(rows);
report.Seconds(1)=report.Seconds(1)+sum(hessianReport.Seconds);
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
if ~converged
    fprintf('Center solve stopped early; fixed geometry is valid but its efficiency still needs diagnostics.\n');
end
end
function [chain,audit]=sampleChain(R,start,cfg)
t0=tic; x=start; chain=zeros(cfg.chainDraws,R.nVar); minWidth=inf; maxWidth=0;
failures=0; minStep=inf; maxResidual=0; maxEq=0; maxLoss=0;
[sr,er,lr]=residuals(x,R);
sinceReport=0;
if isfield(R,'DirectionBasis'), directionBasis=R.DirectionBasis; else, directionBasis=R.B; end
for t=1:(cfg.burnIn+cfg.chainDraws)
    if strcmp(cfg.directionStrategy,'coordinate_mixture') && ...
            (cfg.sphereProbability==0 || rand>cfg.sphereProbability)
        j=randi(size(directionBasis,2)); d=directionBasis(:,j);
    else
        v=randn(size(directionBasis,2),1); d=directionBasis*v;
    end
    d=d/norm(d);
    [left,right]=chord(x,d,R,cfg); width=right-left;
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
    if t>cfg.burnIn, chain(t-cfg.burnIn,:)=x'; end
    if isfield(cfg,'queue') && ~isempty(cfg.queue)
       sinceReport=sinceReport+1;
       if sinceReport>=cfg.progressEvery || t==cfg.burnIn+cfg.chainDraws
           send(cfg.queue,sinceReport); sinceReport=0;
       end
   end
end
audit=struct('Seconds',toc(t0),'Transitions',cfg.burnIn+cfg.chainDraws,'BurnIn',cfg.burnIn, ...
    'DiagnosticThin',1,'ContinuousDraws',cfg.chainDraws,'NumericalFailures',failures,'MinChord',minWidth, ...
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
    left=bisectLine(left,lo,up,lod,upd,R.cap,cfg.lineIterations);
end
if lineLoss(right,lo,up,lod,upd)>R.cap
    right=bisectLine(right,lo,up,lod,upd,R.cap,cfg.lineIterations);
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
function [left,right]=chordReference(x,d,R,cfg)
slack=R.sb-R.S*x; assert(all(slack>=-cfg.feasibilityTolerance),'HAR:Outside','Start outside structure.');
slack=max(slack,0); den=R.S*d; left=-inf; right=inf;
positive=den>0; negative=den<0;
if any(positive), right=min(slack(positive)./den(positive)); end
if any(negative), left=max(slack(negative)./den(negative)); end
assert(isfinite(left)&&isfinite(right)&&left<=0&&right>=0,'HAR:Chord','Unbounded/invalid structural chord.');
if isempty(R.y), return; end
if lossValue(x+left*d,R)>R.cap, left=bisectEndpoint(left,x,d,R,cfg); end
if lossValue(x+right*d,R)>R.cap, right=bisectEndpoint(right,x,d,R,cfg); end
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
%% 7. MCMC diagnostics
function [D,S,C,pred]=analyseChains(chain,R,P,ids,cfg)
[N,~,M]=size(chain); n=size(P,1); U=zeros(N,n,M);
parameterOnly=isfield(cfg,'parameterOnly')&&cfg.parameterOnly;
if parameterOnly, pred=[]; else, pred=zeros(N,n,M); end
for c=1:M
    U(:,:,c)=chain(:,1:R.p,c)*P';
    if ~parameterOnly
        pc=ones(N,n);
        for h=1:R.q-1, pc=pc+(U(:,:,c)>chain(:,R.p+h,c)); end
        pred(:,:,c)=pc;
    end % equality belongs to the lower class; no label epsilon here
end
labels=["theta_"+string(1:R.p),"threshold_"+string(1:R.q-1),"U_object_"+string(ids')];
monitored=cat(2,chain,U); paramCells=cell(size(monitored,2),1);
for j=1:size(monitored,2)
    stat=diagnoseScalar(reshape(monitored(:,j,:),N,M));
    stat.Variable=labels(j); paramCells{j}=stat;
end
D=struct2table(vertcat(paramCells{:}));
if isfield(cfg,'parameterOnly')&&cfg.parameterOnly
    S=table(); C=table(); pred=[]; return;
end
supportCells=cell(n,1);
for i=1:n
    localRows=struct([]);
    for h=1:R.q
        z=double(reshape(pred(:,i,:)==h,N,M)); stat=diagnoseScalar(z);
        stat.ObjectID=ids(i); stat.Class=h; stat.Support=mean(z(:));
        stat.ChainRange=max(mean(z,1))-min(mean(z,1));
        localRows=appendStruct(localRows,stat);
    end
    supportCells{i}=localRows(:);
end
S=struct2table(vertcat(supportCells{:}));
if isfield(cfg,'computeCheckpoints')&&~cfg.computeCheckpoints
    C=table(); return;
end
fractions=[0.1,0.25,0.5,1]; rows=struct([]);
for k=1:numel(fractions)
    stop=max(100,floor(N*fractions(k))); stop=min(stop,N); maxR=NaN; minE=NaN; maxMC=NaN;
    rs=[];es=[];
    for j=1:size(monitored,2)
        d=diagnoseScalar(reshape(monitored(1:stop,j,:),stop,M));
        if d.Status=="ok", rs(end+1)=d.Rhat; es(end+1)=d.BulkESS; end
    end
    maxR=finiteMax(rs); minE=finiteMin(es); ms=[];
    for i=1:n
        for h=1:R.q
            d=diagnoseScalar(double(reshape(pred(1:stop,i,:)==h,stop,M)));
            if d.Status=="ok", ms(end+1)=d.MCSE; end
        end
    end
    maxMC=finiteMax(ms);
    row=struct('RetainedPerChain',stop,'TotalRetained',stop*M, ...
        'MaxRhat',maxR,'MinBulkESS',minE,'MaxNonconstantSupportMCSE',maxMC);
    rows=appendStruct(rows,row);
end
C=struct2table(rows);
end
function stat=diagnoseScalar(x)
stat=struct('Mean',mean(x(:)),'SD',std(x(:)),'Rhat',NaN,'BulkESS',NaN, ...
    'TailESS',NaN,'MeanESS',NaN,'MCSE',NaN,'Status',"ok");
if max(x(:))==min(x(:)), stat.Status="constant_unverified"; return; end
n=floor(size(x,1)/2); split=[x(1:n,:),x(end-n+1:end,:)];
ranked=rankNormal(split); folded=rankNormal(abs(split-median(split(:))));
stat.Rhat=max(splitRhat(ranked),splitRhat(folded));
stat.BulkESS=multiESS(ranked); lo=empiricalQuantile(split(:),0.05); hi=empiricalQuantile(split(:),0.95);
stat.TailESS=min(multiESS(double(split<=lo)),multiESS(double(split>=hi)));
stat.MeanESS=multiESS(split);
if isfinite(stat.MeanESS), stat.MCSE=std(x(:))/sqrt(stat.MeanESS); end
end
function z=rankNormal(x)
ranks=pooledRanks(x);
u=(ranks-3/8)/(numel(x)+1/4); z=sqrt(2)*erfinv(2*u-1);
end
function ranks=pooledRanks(x)
[v,order]=sort(x(:));
first=[1;find(diff(v)~=0)+1]; last=[first(2:end)-1;numel(v)];
tiedRanks=repelem((first+last)/2,last-first+1);
ranks=zeros(size(v)); ranks(order)=tiedRanks;
ranks=reshape(ranks,size(x));
end
function r=splitRhat(x)
n=size(x,1); W=mean(var(x,0,1)); between=n*var(mean(x,1),0);
if W==0
    if between==0, r=NaN; else, r=Inf; end
else, r=sqrt(((n-1)/n*W+between/n)/W); end
end
function ess=multiESS(x)
[n,m]=size(x); W=mean(var(x,0,1)); vp=(n-1)/n*W+var(mean(x,1),0);
if vp<=0||W==0, ess=NaN; return; end
centered=x-mean(x,1); nf=2^nextpow2(2*n);
ft=fft(centered,nf,1); ac=real(ifft(abs(ft).^2,[],1)); ac=ac(1:n,:)/n;
rho=1-(W-mean(ac,2))/vp; rho(1)=1;
pair=[];
for i=1:2:n-1
    value=rho(i)+rho(i+1); if value<=0, break; end
    if ~isempty(pair), value=min(value,pair(end)); end
    pair(end+1)=value;
end
tau=max(1,-1+2*sum(pair)); ess=min(n*m,n*m/tau); % conservative cap
end
function q=empiricalQuantile(v,p)
v=sort(v(:)); index=1+(numel(v)-1)*p; a=floor(index); b=ceil(index);
q=v(a)+(index-a)*(v(b)-v(a));
end
%% 9. Utilities and tests
function validateConfig(c)
assert(any(strcmp(c.mode,{'diagnose','selftest'})),'Invalid mode.');
assert(any(strcmp(c.phase,{'pilot','full'})),'Invalid phase.');
for name={'Ns','chainDraws','burnIn','nChains','L','refPerClass','poolCount', ...
        'lineIterations','pilotRounds','pilotNs','pilotBurnIn','barrierMaxIterations'}
    v=c.(name{1}); assert(isscalar(v)&&isfinite(v)&&v>=1&&v==fix(v),'Invalid %s',name{1});
end
assert(c.Ns==2000&&c.nChains>=4&&mod(c.Ns,c.nChains)==0,'Ns must be 2000, evenly allocated to at least 4 chains.');
assert(c.Ns/c.nChains>=100&&c.chainDraws>=c.Ns/c.nChains ...
    &&mod(c.chainDraws,c.Ns/c.nChains)==0,'chainDraws must be divisible by models per chain.');
assert(c.epsilonPhi>0&&c.epsilonAssignment>0,'Positive tolerances required.');
assert(isa(c.solverTolerance,'double')&&isscalar(c.solverTolerance) ...
    &&isfinite(c.solverTolerance)&&c.solverTolerance>=1e-9&&c.solverTolerance<=1e-3, ...
    'HAR:SolverTolerance', ...
    'Use a double solverTolerance in [1e-9,1e-3] for linprog dual-simplex compatibility.');
assert(c.feasibilityTolerance<=c.epsilonPhi/100,'Feasibility tolerance too large relative to epsilonPhi.');
assert(iscell(c.dataFiles)&&~isempty(c.dataFiles),'Invalid dataset list.');
assert(isscalar(c.usePreconditioning)&&islogical(c.usePreconditioning), ...
    'usePreconditioning must be a logical scalar.');
assert(c.pilotNs>=100&&isscalar(c.covarianceFloor)&&isfinite(c.covarianceFloor) ...
    &&c.covarianceFloor>0&&c.covarianceFloor<=1,'Invalid pilot/covariance settings.');
assert(any(strcmp(c.preconditioningMethod,{'hessian','covariance','hybrid'})), ...
    'preconditioningMethod must be hessian, covariance or hybrid.');
assert(any(strcmp(c.directionStrategy,{'sphere','coordinate_mixture'})), ...
    'directionStrategy must be sphere or coordinate_mixture.');
assert(isscalar(c.sphereProbability)&&isfinite(c.sphereProbability) ...
    &&c.sphereProbability>=0&&c.sphereProbability<=1,'Invalid sphereProbability.');
assert(c.barrierNewtonTolerance>0&&c.barrierArmijo>0&&c.barrierArmijo<0.5 ...
    &&c.barrierStepFraction>0&&c.barrierStepFraction<1,'Invalid barrier settings.');
end
function value=finiteMax(x)
x=x(~isnan(x)); if isempty(x), value=NaN; else, value=max(x); end
end
function value=finiteMin(x)
x=x(~isnan(x)); if isempty(x), value=NaN; else, value=min(x); end
end
function values=appendStruct(values,row)
if isempty(values)
    values=row;
else
    values(end+1)=orderfields(row,values(1));
end
end
function f=recordFailure(f,name,ME)
report=getReport(ME,'extended','hyperlinks','off');
row=struct('Case',string(name),'Identifier',string(ME.identifier), ...
    'Message',string(ME.message),'Report',string(report));
f=appendStruct(f,row);
end
function T=completeSummary(rows,cfg)
completed=struct([]);
for k=1:numel(cfg.dataFiles)
    [~,name]=fileparts(cfg.dataFiles{k});
    row=struct('Dataset',string(name),'Ns',cfg.Ns,'Chains',cfg.nChains, ...
        'Thin',cfg.chainDraws/(cfg.Ns/cfg.nChains),'ChainDrawsPerChain',cfg.chainDraws, ...
        'ChainMaxRhat',NaN,'ChainMinBulkESS',NaN,'ChainMinTailESS',NaN, ...
        'PoolMaxSupportRhat',NaN,'PoolMaxSupportMCSE',NaN, ...
        'PoolMinSupportESS',NaN,'PoolMaxChainRange',NaN, ...
        'ConstantSupportCells',NaN,'NumericalFailures',NaN, ...
        'TotalSeconds',NaN,'ScreenPass',false);
    for j=1:numel(rows)
        if rows(j).Dataset==string(name), row=rows(j); break; end
    end
    completed=appendStruct(completed,row);
end
T=struct2table(completed);
end
function report=selfTests(cfg)
testCfg=cfg; testCfg.queue=[]; testCfg.burnIn=500; testCfg.chainDraws=4000; testCfg.nChains=4;
box=struct('p',2,'q',1,'nVar',2,'B',eye(2),'S',[-eye(2);eye(2)], ...
    'sb',[0;0;1;1],'Phi',[],'y',[],'epsilon',0,'eq',zeros(0,2), ...
    'eqValue',zeros(0,1),'cap',0);
simplex=box; simplex.p=3; simplex.nVar=3; simplex.B=null(ones(1,3));
simplex.S=-eye(3); simplex.sb=zeros(3,1); simplex.eq=ones(1,3); simplex.eqValue=1;
rows=struct([]);
for id=1:5
    if id==2
        R=simplex; targets=[1/3,1/18];
    else
        R=box; targets=[0.5,1/12];
        if id>=3, R.DirectionBasis=[2,0.8;0,0.7]; end
        testCfg.sphereProbability=cfg.sphereProbability;
        if id==5
            testCfg.directionStrategy='coordinate_mixture'; testCfg.sphereProbability=0;
        elseif id==4, testCfg.directionStrategy='coordinate_mixture';
        else, testCfg.directionStrategy=cfg.directionStrategy; end
    end
    draws=zeros(testCfg.chainDraws,R.nVar,4);
    for c=1:4
        rng(91000+100*id+c,'twister');
        if id~=2, start=[0.1+0.2*c;0.9-0.15*c];
        else, start=[c+1;6-c;2]; start=start/sum(start); end
        [draws(:,:,c),audit]=sampleChain(R,start,testCfg);
        assert(audit.NumericalFailures==0,'Known region numerical failure.');
    end
    pooled=reshape(permute(draws,[1,3,2]),[],R.nVar);
    meanError=max(abs(mean(pooled,1)-targets(1)));
    varianceError=max(abs(var(pooled,1,1)-targets(2)));
    testNames={'uniform_box','uniform_simplex','preconditioned_uniform_box','rounded_coordinate_mixture_box','rounded_pure_coordinate_box'};
    row=struct('Test',string(testNames{id}), ...
        'MeanError',meanError,'VarianceError',varianceError, ...
        'Pass',meanError<0.025&&varianceError<0.012);
    rows=appendStruct(rows,row);
end
rng(90127,'twister');
probe=box; probe.p=1; probe.q=2; probe.Phi=[0.25;0.75];
probe.y=[1;2]; probe.epsilon=1e-6; probe.cap=0.15;
maxChordError=0;
for k=1:100
    x=rand(2,1);
    while lossValue(x,probe)>probe.cap, x=rand(2,1); end
    d=randn(2,1); d=d/norm(d);
    [a0,b0]=chordReference(x,d,probe,cfg);
    [a1,b1]=chord(x,d,probe,cfg);
    maxChordError=max(maxChordError,max(abs([a0-a1,b0-b1])));
    assert(lossValue(x+a1*d,probe)<=probe.cap && ...
        lossValue(x+b1*d,probe)<=probe.cap,'Fast chord endpoint exceeds cap.');
end
rows=appendStruct(rows,struct('Test',"fast_chord_equivalence", ...
    'MeanError',maxChordError,'VarianceError',0,'Pass',maxChordError<1e-10));
assert(isequal(pooledRanks([2;1;2;1;3]),[3.5;1.5;3.5;1.5;5]), ...
    'HAR:RankRegression','Tied-rank calculation changed.');
rng(91827,'twister'); independent=randn(4000,4); shifted=independent+[0,1,2,3];
a=diagnoseScalar(independent); b=diagnoseScalar(shifted); c=diagnoseScalar(ones(4000,4));
row=struct('Test',"diagnostic_sanity",'MeanError',a.Rhat,'VarianceError',b.Rhat, ...
    'Pass',a.Rhat<1.02&&a.BulkESS>1000&&b.Rhat>1.1&&c.Status=="constant_unverified");
rows=appendStruct(rows,row);
probe=appendStruct(struct([]),struct('A',1,'B',2));
probe=appendStruct(probe,struct('B',4,'A',3));
assert(numel(probe)==2&&probe(2).A==3&&probe(2).B==4, ...
    'HAR:StructRegression','Structure-array accumulation failed.');
report=struct2table(rows);
end

