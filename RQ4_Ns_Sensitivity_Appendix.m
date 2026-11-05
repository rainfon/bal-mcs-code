function RQ4_Ns_Sensitivity_Appendix
clc;
cfg=struct();
cfg.dataFiles={'DBS.csv','MPG.csv','CPU.csv','ESL.csv'};
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
inputs=cell(nD,1); seedNames={'DBS','ESL','MPG','CPU'};
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
    [starts,startInfo]=makeStarts(R,seeds,cfg);
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
texPath=fullfile(cfg.outputDir,[mfilename,'.tex']);
writeLatexDiagnostics(results.Summary,texPath);
% APPENDIX ADDITION 1: append an implementation/coverage table only.
% The existing diagnostics table and its numerical values are unchanged.
appendSamplerImplementationTable(texPath,cfg,jobs,cases);
ns_runSensitivity();
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
fprintf(fid,'%s\n','\small');
fprintf(fid,'%s\n','\caption{Hit-and-run diagnostics on four datasets.}');
fprintf(fid,'%s\n','\label{tab:har_diagnostics}');
fprintf(fid,'%s\n','\renewcommand{\arraystretch}{1.12}');
fprintf(fid,'%s\n','\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}');
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
fprintf(fid,'%s\n','\end{tabular*}');
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
function [starts,info]=makeStarts(R,seeds,cfg)
starts=zeros(R.nVar,numel(seeds)); info=struct([]);
for c=1:numel(seeds)
    rng(seeds(c)+17,'twister'); objective=[R.B*randn(size(R.B,2),1);zeros(numel(R.y),1)];
    [sol,~,report]=solveLP(objective,R.lpA,R.lpRhs,R.lpEq,1,R.lpLb,[],cfg);
    starts(:,c)=0.7*R.center+0.3*sol(1:R.nVar);
    assertFeasible(starts(:,c),R,cfg);
    report.DistanceFromCenter=norm(starts(:,c)-R.center); info=appendStruct(info,report);
end
for a=1:size(starts,2)
    for b=a+1:size(starts,2)
        assert(norm(starts(:,a)-starts(:,b))>1e-12,'HAR:Starts','Distinct starts could not be constructed.');
    end
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

function ns_runSensitivity
%% Initialization
cfg=struct();
cfg.dataFiles={'DBS.csv','MPG.csv','CPU.csv','ESL.csv'};
cfg.scriptDir=fileparts(mfilename('fullpath'));
if isempty(cfg.scriptDir), cfg.scriptDir=pwd; end
cfg.dataDir=fullfile(cfg.scriptDir,'Datasets');
if ~isfolder(cfg.dataDir), cfg.dataDir=cfg.scriptDir; end
cfg.outputDir=cfg.scriptDir;
cfg.poolCount=50;
cfg.refPerClass=3;
cfg.numRepeats=30;
cfg.L=3;
cfg.NsValues=[500 1000 2000 5000];
% Appendix-only deterministic metrics; these do not change sampling.
cfg.appendixEtaSigma=2;
cfg.appendixXi=0.4;
cfg.epsilonAssignment=1e-6;
cfg.epsilonPhi=1e-5;
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
cfg.useMexSampler=true;
cfg.verbose=false;
cfg.displayDetails=false;
cfg.progressEvery=1000;
cfg.queue=[];
cfg.splitSeed=20261009;
cfg.chainSeed=271828;
cfg.seedNames={'DBS','ESL','MPG','CPU'};

%% Validation and Inputs
assert(numel(cfg.dataFiles)==4,'Exactly four datasets are required.');
assert(exist('linprog','file')==2,'Optimization Toolbox is required.');
assert(cfg.numRepeats>=2 && cfg.nChains>=2);
allCounts=cfg.NsValues;
assert(all(allCounts>0) && all(mod(allCounts,cfg.nChains)==0));
assert(all(mod(cfg.chainDraws,allCounts/cfg.nChains)==0), ...
    'Each per-chain retained count must divide chainDraws.');
assert(cfg.epsilonPhi>0);
% A common storage grid reproduces exactly every requested thinning schedule.
thins=cfg.chainDraws./(cfg.NsValues/cfg.nChains);
cfg.storageThin=thins(1);
for n=2:numel(thins), cfg.storageThin=gcd(cfg.storageThin,thins(n)); end
cfg.storageNs=cfg.nChains*cfg.chainDraws/cfg.storageThin;
addpath(cfg.scriptDir);
if cfg.useMexSampler
    assert(exist('har_chord_mex','file')==3, ...
        'Place har_chord_mex beside this script, or set cfg.useMexSampler=false.');
end
oldRng=rng; rngGuard=onCleanup(@()rng(oldRng));
inputs=cell(4,1);
for d=1:4
    dataPath=fullfile(cfg.dataDir,cfg.dataFiles{d});
    assert(exist(dataPath,'file')==2,'Missing CSV: %s',dataPath);
    [X,y,preprocessing]=ns_readData(dataPath);
    [~,name]=fileparts(dataPath);
    [ref,pool,test]=ns_splitData(y,cfg.refPerClass,cfg.poolCount);
    fprintf('%s: N=%d, classes=%d, reference=%d, initial pool=%d, fixed test=%d.\n', ...
        name,numel(y),max(y),numel(ref),numel(pool),numel(test));
    inputs{d}=struct('X',X,'y',y,'Dataset',name,'Preprocessing',preprocessing);
end

%% Execution
queue=[];
if cfg.useParallel
    assert(license('test','Distrib_Computing_Toolbox'),'Parallel Computing Toolbox required.');
    pool=gcp('nocreate');
    if isempty(pool)
        cluster=parcluster('local');
        cluster.NumWorkers=max(cluster.NumWorkers,cfg.workers);
        pool=parpool(cluster,cfg.workers);
    end
    if cfg.useMexSampler, addAttachedFiles(pool,{which('har_chord_mex')}); end
    queue=parallel.pool.DataQueue;
end
nD=4; nS=numel(cfg.NsValues); nJ=nD*cfg.numRepeats;
totalSteps=nJ*nS;
ns_progressReport([],totalSteps);
if cfg.useParallel, afterEach(queue,@(m)ns_progressReport(m,totalSteps)); end
wall=tic; jobs=cell(nJ,1);
if cfg.useParallel
    parfor j=1:nJ
        d=floor((j-1)/cfg.numRepeats)+1; r=mod(j-1,cfg.numRepeats)+1;
        jobs{j}=ns_runStabilityRepeat(inputs{d},d,r,cfg,queue,totalSteps);
    end
else
    for j=1:nJ
        d=floor((j-1)/cfg.numRepeats)+1; r=mod(j-1,cfg.numRepeats)+1;
        jobs{j}=ns_runStabilityRepeat(inputs{d},d,r,cfg,queue,totalSteps);
    end
end

%% Results
results=struct();
results.DatasetNames=cellfun(@(x)x.Dataset,inputs,'UniformOutput',false);
results.NsValues=cfg.NsValues; results.Jobs=jobs;
results.SupportTV=zeros(cfg.numRepeats,nS,nD);
for j=1:nJ
    d=floor((j-1)/cfg.numRepeats)+1; r=mod(j-1,cfg.numRepeats)+1;
    results.SupportTV(r,:,d)=jobs{j}.SupportTV;
end
v=results.SupportTV;
assert(all(isfinite(v(:))) && all(v(:)>=0 & v(:)<=1));
results.WallSeconds=toc(wall);
appendNsConsistencyTable(fullfile(cfg.outputDir,[mfilename,'.tex']), ...
    results,cfg);
end

%% Paired Sampling at a Fixed Initial State
function job=ns_runStabilityRepeat(input,d,r,cfg,queue,totalSteps)
seedID=find(strcmpi(input.Dataset,cfg.seedNames),1);
assert(~isempty(seedID),'Unknown dataset seed ID.');
rng(cfg.splitSeed+seedID,'twister');
[ref,pool,test]=ns_splitData(input.y,cfg.refPerClass,cfg.poolCount);
y=input.y; q=max(y); P=ns_basis(input.X,cfg.L);
[R,~]=ns_makeRegion(input.X(ref,:),y(ref),q,cfg);
key=sprintf('dataset=%d;repeat=%d;',d,r);
pilotSeeds=ns_groupSeeds(cfg,d,r,[key 'pilot']);
[pilotStarts,~]=ns_makeStarts(R,pilotSeeds,cfg);
[R,~]=ns_preparePreconditioner(R,pilotStarts,pilotSeeds,cfg);
seedsA=ns_groupSeeds(cfg,d,r,[key 'A']);
if r==1, seedsA=cfg.chainSeed+10000*seedID+100+(1:cfg.nChains); end
seedsB=ns_groupSeeds(cfg,d,r,[key 'B']);
assert(numel(unique([pilotSeeds seedsA seedsB]))==3*cfg.nChains, ...
    'Seed collision between independent sampling groups.');
[startsA,~]=ns_makeStarts(R,seedsA,cfg);
[startsB,~]=ns_makeStarts(R,seedsB,cfg);
modelsA=ns_sampleGroup(R,startsA,seedsA,cfg.storageNs,cfg);
modelsB=ns_sampleGroup(R,startsB,seedsB,cfg.storageNs,cfg);
nS=numel(cfg.NsValues);
job=struct('Dataset',input.Dataset,'Repeat',r,'ReferenceIDs',ref, ...
    'PoolIDs',pool,'TestIDs',test,'SeedsA',seedsA,'SeedsB',seedsB, ...
    'SupportTV',zeros(1,nS), ...
    'RecommendationAgreement',zeros(1,nS), ...
    'RegionAgreement',zeros(1,nS), ...
    'TestOMAE',zeros(1,nS), ...
    'OMAEGroupDifference',zeros(1,nS));
for n=1:nS
    M1=ns_selectRetained(modelsA,cfg.NsValues(n),cfg);
    M2=ns_selectRetained(modelsB,cfg.NsValues(n),cfg);
    ms1=ns_estimateSupport(M1,R,P(pool,:));
    ms2=ns_estimateSupport(M2,R,P(pool,:));
    job.SupportTV(n)=mean(0.5*sum(abs(ms1-ms2),2));
    % APPENDIX ADDITION 2: analyze the same retained samples, without
    % new transitions, random draws, or changes to the SupportTV result.
    [recA,regionA,omaeA]=ns_appendixDecisions(M1,R,P,pool,ref,test,y,cfg,ms1);
    [recB,regionB,omaeB]=ns_appendixDecisions(M2,R,P,pool,ref,test,y,cfg,ms2);
    job.RecommendationAgreement(n)=mean(recA==recB);
    job.RegionAgreement(n)=mean(regionA==regionB);
    job.TestOMAE(n)=0.5*(omaeA+omaeB);
    job.OMAEGroupDifference(n)=abs(omaeA-omaeB);
    message=struct('Kind','step','Count',1);
    if isempty(queue), ns_progressReport(message,totalSteps); else, send(queue,message); end
end
end

%% Independent Chain Groups
function seeds=ns_groupSeeds(cfg,d,r,key)
base=ns_keySeed(cfg.chainSeed,d,r,key);
seeds=zeros(1,cfg.nChains);
for c=1:cfg.nChains, seeds(c)=ns_keySeed(base,0,c,'chain'); end
end

function M=ns_sampleGroup(R,starts,seeds,Ns,cfg)
localCfg=cfg; localCfg.Ns=Ns; localCfg.queue=[];
perChain=Ns/cfg.nChains; M=zeros(R.nVar,Ns);
for c=1:cfg.nChains
    rng(seeds(c),'twister');
    [chain,audit]=ns_sampleChain(R,starts(:,c),localCfg);
    assert(audit.NumericalFailures==0,'NsStability:NumericalFailure', ...
        'Sampler numerical failure; inspect geometry.');
    assert(size(chain,1)==perChain);
    M(:,(c-1)*perChain+(1:perChain))=chain';
end
end

%% Exact Thinning Subsets
function M=ns_selectRetained(stored,Ns,cfg)
storedPerChain=cfg.storageNs/cfg.nChains;
targetPerChain=Ns/cfg.nChains;
stride=storedPerChain/targetPerChain;
assert(stride==fix(stride));
columns=zeros(1,Ns);
for c=1:cfg.nChains
    columns((c-1)*targetPerChain+(1:targetPerChain))= ...
        (c-1)*storedPerChain+(stride:stride:storedPerChain);
end
M=stored(:,columns);
end

%% Class Support
function ms=ns_estimateSupport(M,R,P)
U=P*M(1:R.p,:); assignment=ones(size(U));
for h=1:R.q-1, assignment=assignment+(U>M(R.p+h,:)); end
ms=zeros(size(U,1),R.q);
for h=1:R.q, ms(:,h)=mean(assignment==h,2); end
assert(max(abs(sum(ms,2)-1))<1e-10);
end

%% Progress
function ns_progressReport(message,total)
persistent started completed lastPercent
if isempty(message)
    started=tic; completed=0; lastPercent=-1;
    fprintf('Progress: 0%% | Elapsed: 00:00:00 | Remaining: --:--:--\n');
    return;
end
completed=completed+message.Count;
percent=min(100,floor(100*completed/max(total,1)));
if percent==lastPercent, return; end
elapsed=toc(started);
if completed>0
    remaining=formatDuration(elapsed*(total-completed)/completed);
else
    remaining='--:--:--';
end
fprintf('Progress: %d%% | Elapsed: %s | Remaining: %s\n', ...
    percent,formatDuration(elapsed),remaining);
lastPercent=percent;
end

function seed=ns_keySeed(base,d,r,key)
prime=2147483647;
seed=mod(double(base)+104729*d+13007*r,prime);
bytes=double(char(key));
for j=1:numel(bytes), seed=mod(seed*131+bytes(j),prime); end
seed=max(1,floor(seed));
end
%% Read Data
function [X,y,info]=ns_readData(file)
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
function [ref,pool,test]=ns_splitData(y,nref,npool)
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
function P=ns_basis(X,L)
P=zeros(size(X,1),size(X,2)*L);
for j=1:size(X,2)
    for k=1:L, P(:,(j-1)*L+k)=max(0,min(1,L*X(:,j)-(k-1))); end
end
end
%% Make Region
function [R,info]=ns_makeRegion(X,y,q,cfg)
P=ns_basis(X,cfg.L); p=size(P,2); n=numel(y); nv=p+q-1;
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
[sol,fval,solver]=ns_solveLP(f,A,rhs,Aeq,1,lb,[],cfg);
R=struct('p',p,'q',q,'nVar',nv,'B',B,'S',S,'sb',sb,'Phi',P, ...
    'y',y,'epsilon',cfg.epsilonAssignment,'eq',eq,'eqValue',1);
actual=ns_lossValue(sol(1:nv),R);
R.phiStar=max([0,fval,actual]); R.cap=R.phiStar+cfg.epsilonPhi;
capRow=[zeros(1,nv),ones(1,n)]; A=[A;capRow]; rhs=[rhs;R.cap];
norms=[sqrt(sum((S*B).^2,2));sqrt(sum((lossRows*B).^2,2));0];
ballA=[A,norms]; ballEq=[Aeq,0]; ballF=[zeros(nv+n,1);-1];
[ball,~,centerSolver]=ns_solveLP(ballF,ballA,rhs,ballEq,1,[lb;0],[],cfg);
R.center=ball(1:nv); R.radius=ball(end);
assert(R.radius>cfg.minimumRadius,'HAR:Degenerate', ...
    'No certified full-relative-dimensional ball. Do not increase epsilon silently; inspect LP/affine hull.');
ns_assertFeasible(R.center,R,cfg);
R.lpA=A; R.lpRhs=rhs; R.lpEq=Aeq; R.lpLb=lb;
info=struct('MinimumLossObjective',fval,'ActualLossAtOptimum',actual, ...
    'NumericalLossGap',actual-fval,'MinimumLossSolver',solver,'CenterSolver',centerSolver, ...
    'BallConstruction','Conservative sum of per-object worst-branch bounds');
end
%% Solve Lp
function [x,fval,report]=ns_solveLP(f,A,b,Aeq,beq,lb,ub,cfg)
persistent defaultOptions optionsTolerance
if isempty(defaultOptions) || ~isequal(optionsTolerance,cfg.solverTolerance)
    defaultOptions=optimoptions('linprog','Display','none', ...
        'ConstraintTolerance',cfg.solverTolerance,'OptimalityTolerance',cfg.solverTolerance);
    optionsTolerance=cfg.solverTolerance;
end
opts=defaultOptions;
As=sparse(A); Es=sparse(Aeq);
[x,fval,flag,output]=linprog(f,As,b,Es,beq,lb,ub,opts);
res=ns_lpPrimalResidual(x,As,b,Es,beq,lb,ub);
accepted=ns_lpSolutionAccepted(x,fval,flag,res,cfg);
attempts={struct('Algorithm',opts.Algorithm,'ExitFlag',flag, ...
    'MaxPrimalResidual',res,'Output',output)};
if accepted
    report=ns_lpReport(flag,output,res,opts.Algorithm,attempts);
    return;
end
originalRng=rng; rngCleanup=onCleanup(@()rng(originalRng));
warning('Comparison:LPRetry', ...
    ['LP failed validation (exitflag=%d, residual=%.3g). Retrying the same LP ', ...
     'with MaxIterations=%d. Original message: %s'], ...
    flag,res,cfg.lpRetryMaxIterations,output.message);
retry=optimoptions(opts,'MaxIterations',cfg.lpRetryMaxIterations);
[x,fval,flag,output]=linprog(f,As,b,Es,beq,lb,ub,retry);
res=ns_lpPrimalResidual(x,As,b,Es,beq,lb,ub);
attempts{end+1}=struct('Algorithm',retry.Algorithm,'ExitFlag',flag, ...
    'MaxPrimalResidual',res,'Output',output);
if ns_lpSolutionAccepted(x,fval,flag,res,cfg)
    report=ns_lpReport(flag,output,res,retry.Algorithm,attempts);
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
res=ns_lpPrimalResidual(x,As,b,Es,beq,lb,ub);
attempts{end+1}=struct('Algorithm',fallback.Algorithm,'ExitFlag',flag, ...
    'MaxPrimalResidual',res,'Output',output);
if ~ns_lpSolutionAccepted(x,fval,flag,res,cfg)
    error('HAR:LP', ...
        ['LP failed after default solver, increased-iteration retry and %s fallback. ', ...
         'Final exitflag=%d; original-constraint residual=%.3g (limit %.3g). %s'], ...
        fallback.Algorithm,flag,res,cfg.feasibilityTolerance,output.message);
end
report=ns_lpReport(flag,output,res,fallback.Algorithm,attempts);
end
%% Lp Primal Residual
function res=ns_lpPrimalResidual(x,A,b,Aeq,beq,lb,ub)
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
function ok=ns_lpSolutionAccepted(x,fval,flag,res,cfg)
ok=flag>0 && ~isempty(x) && all(isfinite(x)) && ...
    isscalar(fval) && isfinite(fval) && isfinite(res) && res<=cfg.feasibilityTolerance;
end
%% Lp Report
function report=ns_lpReport(flag,output,res,algorithm,attempts)
report=struct('ExitFlag',flag,'Message',output.message, ...
    'MaxPrimalResidual',res,'Algorithm',algorithm, ...
    'Retried',numel(attempts)>1,'Attempts',{attempts});
end
%% Make Starts
function [starts,info]=ns_makeStarts(R,seeds,cfg)
starts=zeros(R.nVar,numel(seeds)); info=struct([]);
for c=1:numel(seeds)
    rng(seeds(c)+17,'twister'); objective=[R.B*randn(size(R.B,2),1);zeros(numel(R.y),1)];
    [sol,~,report]=ns_solveLP(objective,R.lpA,R.lpRhs,R.lpEq,1,R.lpLb,[],cfg);
    starts(:,c)=0.7*R.center+0.3*sol(1:R.nVar);
    ns_assertFeasible(starts(:,c),R,cfg);
    report.DistanceFromCenter=norm(starts(:,c)-R.center); info=ns_appendStruct(info,report);
end
end
%% Prepare Preconditioner
function [R,report]=ns_preparePreconditioner(R,starts,seeds,cfg)
assert(strcmp(cfg.preconditioningMethod,'hessian'),'Only fixed Hessian geometry is supported.');
if cfg.usePreconditioning
    [R,report]=ns_prepareHessianGeometry(R,cfg);
else
    R.DirectionBasis=R.B; report=table();
end
end
%% Prepare Hessian Geometry
function [R,report]=ns_prepareHessianGeometry(R,cfg)
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
center=x0+R.B*z; ns_assertFeasible(center,R,cfg);
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
function [chain,audit]=ns_sampleChain(R,start,cfg)
t0=tic; x=start;
burnIn=cfg.burnIn; chainDraws=cfg.chainDraws; nTrans=burnIn+chainDraws;
keepThin=chainDraws/(cfg.Ns/cfg.nChains);
chain=zeros(cfg.Ns/cfg.nChains,R.nVar); kept=0;
minWidth=inf; maxWidth=0;
failures=0; minStep=inf; maxResidual=0; maxEq=0; maxLoss=0;
[sr,er,lr]=ns_residuals(x,R);
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
                left=ns_affineEndpoint(left,lo,up,lod,upd,cap,lineIterations);
            end
            if sum(max(0,max(lo+right*lod,up+right*upd)))>cap
                right=ns_affineEndpoint(right,lo,up,lod,upd,cap,lineIterations);
            end
            xL=x+left*d; UL=Phi*xL(1:p); bL=xL(p+1:end);
            lower(:)=0; upper(:)=0;
            lower(yGt1)=bL(yLo)+epsilon-UL(yGt1); upper(yLtQ)=UL(yLtQ)-bL(yUp);
            if sum(max(0,max(lower,upper)))>cap
                left=ns_bisectEndpoint(originalLeft,x,d,R,cfg);
            end
            xR=x+right*d; UR=Phi*xR(1:p); bR=xR(p+1:end);
            lower(:)=0; upper(:)=0;
            lower(yGt1)=bR(yLo)+epsilon-UR(yGt1); upper(yLtQ)=UR(yLtQ)-bR(yUp);
            if sum(max(0,max(lower,upper)))>cap
                right=ns_bisectEndpoint(originalRight,x,d,R,cfg);
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
                packet=cfg.progressContext; packet.Count=sinceReport; send(cfg.queue,packet);
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
function value=ns_lineLoss(t,lo,up,lod,upd)
value=sum(max(0,max(lo+t*lod,up+t*upd)));
end
%% Affine Endpoint
function endpoint=ns_affineEndpoint(outside,lo,up,lod,upd,cap,iterations)
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
endpoint=ns_bisectLine(original,lo,up,lod,upd,cap,iterations);
end
%% Bisect Line
function endpoint=ns_bisectLine(outside,lo,up,lod,upd,cap,iterations)
inside=0;
for k=1:iterations
    middle=0.5*(outside+inside);
    if middle==outside||middle==inside, break; end
    if ns_lineLoss(middle,lo,up,lod,upd)<=cap
        inside=middle;
    else
        outside=middle;
    end
end
endpoint=inside;
end
%% Bisect Endpoint
function endpoint=ns_bisectEndpoint(outside,x,d,R,cfg)
inside=0;
assert(ns_lossValue(x,R)<=R.cap,'HAR:LossStart','Current point exceeds exact loss cap.');
for k=1:cfg.lineIterations
    middle=0.5*(outside+inside);
    if middle==outside||middle==inside, break; end
    if ns_lossValue(x+middle*d,R)<=R.cap, inside=middle; else, outside=middle; end
end
endpoint=inside;
end
%% Loss Value
function value=ns_lossValue(x,R)
if isempty(R.y), value=0; return; end
U=R.Phi*x(1:R.p); b=x(R.p+1:end); lower=zeros(numel(R.y),1); upper=lower;
ids=R.y>1; lower(ids)=b(R.y(ids)-1)+R.epsilon-U(ids);
ids=R.y<R.q; upper(ids)=U(ids)-b(R.y(ids));
value=sum(max(0,max(lower,upper)));
end
%% Residuals
function [s,e,l]=ns_residuals(x,R)
s=max([0;R.S*x-R.sb]); e=max([0;abs(R.eq*x-R.eqValue)]); l=max(0,ns_lossValue(x,R)-R.cap);
end
%% Assert Feasible
function ns_assertFeasible(x,R,cfg)
[s,e,l]=ns_residuals(x,R);
assert(all(isfinite(x))&&max([s,e,l])<=cfg.feasibilityTolerance, ...
    'HAR:Feasibility','Invalid parameter: structural=%g equality=%g loss=%g',s,e,l);
end
%% Pooled Ranks
function q=ns_empiricalQuantile(v,p)
v=sort(v(:)); index=1+(numel(v)-1)*p; a=floor(index); b=ceil(index);
q=v(a)+(index-a)*(v(b)-v(a));
end
%% Append Struct
function values=ns_appendStruct(values,row)
if isempty(values)
    values=row;
else
    values(end+1)=orderfields(row,values(1));
end
end


%% APPENDIX ONLY: Implementation table (appended after the original TeX table)
function appendSamplerImplementationTable(path,cfg,jobs,cases)
% The table documents the actual implementation, not a new diagnostic run.
% Existing audit fields already collected by sampleChain are aggregated.
maxResidual=0; allFailures=0; smallestStartDistance=inf;
for di=1:numel(cases)
    starts=cases{di}.Starts;
    for a=1:size(starts,2)
        for b=a+1:size(starts,2)
            smallestStartDistance=min(smallestStartDistance, ...
                norm(starts(:,a)-starts(:,b)));
        end
    end
end
for di=1:numel(jobs)
    if isempty(jobs{di}) || ~isfield(jobs{di},'Audit'), continue; end
    a=jobs{di}.Audit;
    maxResidual=max(maxResidual,max([a.MaxStructuralResidual; ...
        a.MaxEqualityResidual;a.MaxLossResidual]));
    allFailures=allFailures+sum(a.NumericalFailures);
end
fid=fopen(path,'a');
assert(fid>0,'Appendix:TeX','Cannot append to %s',path);
cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
eol=[char(92),char(92)];
fprintf(fid,'\n%% APPENDIX ADDITION: original HAR diagnostics above unchanged.\n');
fprintf(fid,'\\begin{table}[htbp]\n\\centering\n\\small\n');
fprintf(fid,'\\caption{Implementation of model-space hit-and-run and scope of the reported diagnostics.}\n');
fprintf(fid,'\\label{tab:har_implementation}\n');
fprintf(fid,'\\renewcommand{\\arraystretch}{1.16}\n');
fprintf(fid,'\\begin{tabular}{@{}lp{0.75\\linewidth}@{}}\n\\toprule\n');
fprintf(fid,'Step & Implemented procedure %s\n\\midrule\n',eol);
fprintf(fid,'Target & Direct parameter vector $x=(\\boldsymbol{\\theta},\\mathbf b)$; LP slack variables are used only to obtain $\\Phi^*$ and a feasible interior point, not sampled. %s\n',eol);
fprintf(fid,'Affine directions & The null-space basis of $\\sum\\theta=1$ is used. Fixed Hessian preconditioning; %.0f\\%% preconditioned coordinate directions and %.0f\\%% transformed Gaussian directions. %s\n',100*(1-cfg.sphereProbability),100*cfg.sphereProbability,eol);
fprintf(fid,'Feasible chord & Intersect the structural bounds $S(x+td)\\leq s_b$ with $L_D(x+td)\\leq\\Phi^*+\\varepsilon_\\Phi$, using piecewise-linear loss evaluation and constrained endpoint search. %s\n',eol);
fprintf(fid,'Transition & Draw $u\\sim\\mathrm{Uniform}(0,1)$, then set $x^+=x+[t_-+u(t_+-t_-)]d$. No direction redraw upon a rejected proposal. %s\n',eol);
fprintf(fid,'Four starts & For each chain, solve a randomly directed LP and use $x_0^{(c)}=0.7x_{\\rm center}+0.3x_{\\rm LP}^{(c)}$, with independent random seeds and a feasibility check (minimum observed starting-point separation: $%.2e$). %s\n',smallestStartDistance,eol);
fprintf(fid,'Numerics & Solver and feasibility tolerances are both $%.0e$; chord endpoint search uses %d iterations. Widths $\\leq64\\,\\mathrm{eps}(\\max\\{1,\\lVert x\\rVert\\})$ are retained as self-loops and counted together with rejected candidates. %s\n',cfg.feasibilityTolerance,cfg.lineIterations,eol);
fprintf(fid,'Audit & Across the four initial-state datasets, the largest recorded structural/equality/loss residual is $%.2e$; the total recorded self-loops/rejections is %d. %s\n',maxResidual,allFailures,eol);
fprintf(fid,'Coverage & Initial information state $D_0$ only (DBS, MPG, CPU, ESL); %d chains, %d burn-in transitions and %d subsequent transitions per chain; %d retained models in the diagnostic table. Updated information states are not examined. %s\n',cfg.nChains,cfg.burnIn,cfg.chainDraws,cfg.Ns,eol);
fprintf(fid,'\\bottomrule\n\\end{tabular}\n');
fprintf(fid,'\\end{table}\n');
end

%% APPENDIX ONLY: decisions from each already-generated sample group
function [recommendation,region,omae]=ns_appendixDecisions(M,R,P,pool,ref,test,y,cfg,ms)
% All results in this function refer to D0 (the fixed initial reference set).
% AVG--TP reconstruction for consistency comparison, with no fresh draws.
model=mean(M,2);
recP=P(pool,:)*model(1:R.p);
recommendation=ones(numel(pool),1);
for h=1:R.q-1
    recommendation=recommendation+(recP>model(R.p+h));
end
predTest=P(test,:)*model(1:R.p);
testClasses=ones(numel(test),1);
for h=1:R.q-1
    testClasses=testClasses+(predTest>model(R.p+h));
end
omae=mean(abs(testClasses-y(test)))/(R.q-1);

% TP distance: mean model-wise L1 distance between per-class
% constraint-violation profiles, as defined in Eqs. (25)--(26).
Ppool=P(pool,:); Pref=P(ref,:);
U=Ppool*M(1:R.p,:);
V=Pref*M(1:R.p,:);
nPool=numel(pool); nRef=numel(ref); nModels=size(M,2);
distance=zeros(nPool,nRef);
for h=1:R.q
    if h>1
        lowerPool=M(R.p+h-1,:)+R.epsilon-U;
        lowerRef =M(R.p+h-1,:)+R.epsilon-V;
    else
        lowerPool=zeros(nPool,nModels);
        lowerRef=zeros(nRef,nModels);
    end
    if h<R.q
        upperPool=U-M(R.p+h,:);
        upperRef=V-M(R.p+h,:);
    else
        upperPool=zeros(nPool,nModels);
        upperRef=zeros(nRef,nModels);
    end
    rhoPool=max(0,max(lowerPool,upperPool));
    rhoRef=max(0,max(lowerRef,upperRef));
    for r=1:nRef
        distance(:,r)=distance(:,r)+mean(abs(rhoPool-rhoRef(r,:)),2);
    end
end
positiveDistance=distance(distance>0);
if isempty(positiveDistance)
    sigma=1e-12;
else
    sigma=max(1e-12,cfg.appendixEtaSigma*median(positiveDistance));
end
weights=exp(-(distance./sigma).^2);
W=sum(weights,2);
local=zeros(nPool,R.q);
for h=1:R.q
    ids=(y(ref)==h);
    if any(ids)
        local(:,h)=sum(weights(:,ids),2)./max(W,realmin);
    end
end
lambda=W./(1+W);
integrated=(1-lambda).*ms+lambda.*local;
% Case W=0 is exact model-space evidence, not an artificial distribution.
noLocal=(W==0);
if any(noLocal), integrated(noLocal,:)=ms(noLocal,:); end

adequacy=1-sum(abs(recommendation-(1:R.q)).*integrated,2)/(R.q-1);
bounds=[0;model(R.p+1:end);1];
left=bounds(recommendation);
right=bounds(recommendation+1);
classWidth=right-left;
mac=ones(nPool,1);
valid=(classWidth>1e-12);
mac(valid)=1-2*min(recP(valid)-left(valid), ...
    right(valid)-recP(valid))./classWidth(valid);
mac=max(0,min(1,mac));
xi=cfg.appendixXi;
alpha=(1-xi)*mac./((1-xi)*mac+xi*(1-mac));
beta=xi*mac./(xi*mac+(1-xi)*(1-mac));
% NEG=1, BND=2, POS=3. Tie priority: POS, then NEG, then BND.
region=2*ones(nPool,1);
region(adequacy<=beta)=1;
region(adequacy>=alpha)=3;
end

%% APPENDIX ONLY: append one numerical consistency table to the same .tex
function appendNsConsistencyTable(path,results,cfg)
nD=numel(results.DatasetNames);
nS=numel(cfg.NsValues);
r=cfg.numRepeats;
rec=zeros(r,nS,nD); reg=rec; omae=rec; diffOmae=rec;
for d=1:nD
    for j=1:r
        job=results.Jobs{(d-1)*r+j};
        rec(j,:,d)=job.RecommendationAgreement;
        reg(j,:,d)=job.RegionAgreement;
        omae(j,:,d)=job.TestOMAE;
        diffOmae(j,:,d)=job.OMAEGroupDifference;
    end
end
fid=fopen(path,'a');
assert(fid>0,'Appendix:TeX','Cannot append to %s',path);
cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
eol=[char(92),char(92)];
fprintf(fid,'\n%% APPENDIX ADDITION: deterministic summaries based on existing samples.\n');
fprintf(fid,'\\begin{table}[htbp]\n\\centering\n\\small\n');
fprintf(fid,'\\caption{Finite-sample agreement of the selected AVG--TP evidence construction at the initial information state.}\n');
fprintf(fid,'\\label{tab:ns_consistency}\n');
fprintf(fid,'\\renewcommand{\\arraystretch}{1.10}\n');
fprintf(fid,'\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}\n\\toprule\n');
fprintf(fid,'Dataset & $N_s$ & Support TV & Rec. agree. (\\%%) & Region agree. (\\%%) & Test OMAE & $|\\Delta\\mathrm{OMAE}|$ %s\n',eol);
fprintf(fid,'\\midrule\n');
for d=1:nD
    for j=1:nS
        fprintf(fid,'%s & %d & %.5f & %.2f & %.2f & %.5f & %.5f %s\n', ...
            results.DatasetNames{d},cfg.NsValues(j), ...
            mean(results.SupportTV(:,j,d)),100*mean(rec(:,j,d)), ...
            100*mean(reg(:,j,d)),mean(omae(:,j,d)), ...
            mean(diffOmae(:,j,d)),eol);
    end
    if d<nD, fprintf(fid,'\\midrule\n'); end
end
fprintf(fid,'\\bottomrule\n\\end{tabular*}\n');
fprintf(fid,'\\vspace{2pt}\n');
fprintf(fid,'\\parbox{\\linewidth}{\\footnotesize Notes: %d paired repetitions per dataset, two independent groups of %d chains, the same fixed $D_0$/pool/test split across sampling repetitions, pool of %d, and $N_s\\in\\{%s\\}$. Support TV is the pre-existing value (unchanged). Agreement measures compare the two groups for the same candidates. Each OMAE is the mean of two AVG-model test-set errors; $|\\Delta\\mathrm{OMAE}|$ is the absolute difference between them. TP similarity uses $\\eta_\\sigma=%.1f$ and the region rule uses $\\xi=%.1f$. The experiment concerns the initial information state only; Test OMAE is not the final OMAE after active-learning updates.}\n', ...
    cfg.numRepeats,cfg.nChains,cfg.poolCount, ...
    strjoin(arrayfun(@num2str,cfg.NsValues,'UniformOutput',false),','), ...
    cfg.appendixEtaSigma,cfg.appendixXi);
fprintf(fid,'\\end{table}\n');
end
