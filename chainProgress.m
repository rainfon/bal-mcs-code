function state=chainProgress(action,packet)
% Tracks the newest chain fraction so a long sampler does not sit at 0%%.
persistent latest shown
if isempty(latest), latest=-1; shown=-1; end
if strcmp(action,'reset')
    latest=-1; shown=-1;
    state=struct('percent',-1,'changed',false);
    return;
end
if strcmp(action,'read')
    state=struct('percent',latest,'changed',false);
    return;
end
state=struct('percent',latest,'changed',false);
if ~isstruct(packet) || ~isfield(packet,'Kind') || ~strcmp(packet.Kind,'transition')
    return;
end
if ~isfield(packet,'Transition') || ~isfield(packet,'TotalTransitions') || packet.TotalTransitions<=0
    return;
end
latest=min(99,floor(100*packet.Transition/packet.TotalTransitions));
state.percent=latest;
state.changed=latest~=shown;
shown=latest;
end
