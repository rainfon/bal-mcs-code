function dirPath=figureDirs(scriptDir)
% Figure PDFs are written under this code folder, not its parent.
% Set BAL_FIGURE_DIR to choose another directory.
override=strtrim(getenv('BAL_FIGURE_DIR'));
if isempty(override)
    dirPath=fullfile(scriptDir,'Figures');
else
    dirPath=override;
end
if ~exist(dirPath,'dir'), mkdir(dirPath); end
end
