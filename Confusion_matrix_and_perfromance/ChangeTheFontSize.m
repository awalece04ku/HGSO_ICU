close all
filename='y_pred_gbc_Confusion_matrix';
openfig([filename,'.fig']);

% Get all text objects in the figure
allTextObjects = findall(gcf,'Type','text');

% Set the font properties of each text object
for i = 1:length(allTextObjects)
    set(allTextObjects(i), 'FontName', 'Times New Roman');
    set(allTextObjects(i), 'FontSize', 15);
    set(allTextObjects(i), 'FontWeight', 'bold');
end

% Get all axes in the figure
allAxes = findall(gcf,'Type','axes');

% Set the font properties of XTickLabel and YTickLabel for each axes
for i = 1:length(allAxes)
    set(allAxes(i).XAxis, 'FontName', 'Times New Roman');
    set(allAxes(i).XAxis, 'FontSize', 13);
    set(allAxes(i).XAxis, 'FontWeight', 'bold');
    
    set(allAxes(i).YAxis, 'FontName', 'Times New Roman');
    set(allAxes(i).YAxis, 'FontSize', 13);
    set(allAxes(i).YAxis, 'FontWeight', 'bold');
end

% Save the modified figure
savefig([filename,'_Modified.fig']);

% Save the modified figure as .eps with 600 dpi
print('-depsc', '-r600', [filename,'.eps']);

% Save the modified figure as .png with 600 dpi
%  print('-dpng', '-r600', [filename,'.png']);

% Save the modified figure as .jpg with 600 dpi
 print('-djpeg', '-r600', [filename,'.jpg']);