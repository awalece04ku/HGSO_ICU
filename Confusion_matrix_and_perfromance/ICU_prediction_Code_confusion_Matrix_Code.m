
clc
clear all
close all
workingFolder = 'C:\Users\uqmawal\Dependencies\';
cd(workingFolder)

test_label = readNPY('y_test.npy');

Predicted_Label_on_test_data = readNPY('y_pred_hgsoxgb.npy');

npylist= {'y_pred_gnb.npy';'y_pred_lda.npy';'y_pred_knn.npy';'y_pred_rf.npy';'y_pred_lgbm.npy';'y_pred_gbc.npy';'y_pred_lr.npy';'y_pred_hgsoxgb.npy'};
%  npylist={'LDA_opt_clf_predict_org.npy';'QLDA_opt_clf_predict_org.npy';'KNN_opt_predict_org.npy';'NB_Opt_clf_predict_org.npy';'DT_opt_clf_predict_org.npy';'RF_opt_clf_predict_org.npy';'XGB_opt_clf_predict_org.npy';'gbcoptclf_predict_org.npy';'SVC_opt_predict_org.npy'};


No_of_npylist=length(npylist);
All_RESULT=[];
for npy=1:No_of_npylist
    
    fname = npylist{npy};
    Predicted_Label = readNPY(fname);
    
   
    
%       test_label = readNPY('test_label_org.npy');
    [c_matrixp,Result]= confusion1.getMatrix(test_label,Predicted_Label,0);
    allRESULT_tmp=[Result.Accuracy Result.Error Result.F1_score Result.FalsePositiveRate Result.Kappa Result.MatthewsCorrelationCoefficient Result.Precision Result.Sensitivity Result.Specificity].*100;
     All_RESULT=[All_RESULT; allRESULT_tmp ];
    
     
     
% figure
h = figure(npy)
confusion_matrix_plotting(test_label+1,Predicted_Label+1);  % so that it start from 1 Not from 0
 hh = gca;

hh.XTickLabel = {'ICU-No','ICU-Yes','Overall'};
hh.XTickLabelRotation = 0;
hh.YTickLabel = {'ICU-No','ICU-Yes','Overall',''};
hh.YTickLabelRotation = 90;
title('')
xlabel('Actual Class')
ylabel('Target class')
[C,k] = strsplit(fname,'.npy');
title('')
savefig(h,[C{1},'_Confusion_matrix.fig'])

% exportgraphics(hh,[C{1},'_Confusion_matrix.png'],'Resolution',1200)

clear hh  C  Predicted_Label
     
     
    
end


Perfromance_Matrics = array2table(All_RESULT,'RowNames',{'y_pred_gnb.npy';'y_pred_lda.npy';'y_pred_knn.npy';'y_pred_rf.npy';'y_pred_lgbm.npy';'y_pred_gbc.npy';'y_pred_lr.npy';'y_pred_hgsoxgb.npy'},...,
    'VariableNames',{'Accuracy','Error','F1_score','FalsePositiveRate','Kappa','MatthewsCorrelationCoefficient','Precision','Sensitivity','Specificity'})
writetable(Perfromance_Matrics,'ICU_Prediction_perfromance_metrics.csv','WriteRowNames', true )
save All_RESULT_org All_RESULT Perfromance_Matrics

