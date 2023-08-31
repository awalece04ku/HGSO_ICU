function [] = roc_plotting(Target,Y_output)
 if any(Target < 1)
      error(message(' MAKE IT POSITVE BEFORE ANALSYSIS'));
 end
Target=Target(:)';
Y_output=Y_output(:)';
M = size(unique(Target),2);
N = size(Target,2);
targets = zeros(M,N);
outputs = zeros(M,N);
targetsIdx = sub2ind(size(targets), Target, 1:N); %linearInd = sub2ind(matrixSize, rowSub, colSub)
outputsIdx = sub2ind(size(outputs), Y_output, 1:N);
targets(targetsIdx) = 1;
outputs(outputsIdx) = 1;
% Plot the confusion matrix
plotroc(targets,outputs)

end