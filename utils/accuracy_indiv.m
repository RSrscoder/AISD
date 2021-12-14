function [TP,TN,FP,FN,sum] = accuracy_indiv(result,GT)
% -------------------------------------------------------------------------
%   Description:
%       Shadow Detection Accuracy Assessment
%
%   Input:
%       - result : shadow detection result
%       - GT : shadow ground truth
%
%   Output:
%       TP :     true positive, the number of true shadow pixels which are identified correctly
%       TN :     true negative, the number of nonshadow pixels which are identified correctly
%       FP :     false positive, the number of nonshadow pixels which are identified as true shadow pixels
%       FN ;     false negative, the number of true shadow pixels which are identified as nonshadow pixels
%       sum:     sum of the image pixels
% -------------------------------------------------------------------------

%% Initial setting
% Get the size of the input image
[m,n] = size(result); 

% Set the original data
TP = 0;     % true positive, the number of true shadow pixels which are identified correctly
TN = 0;     % true negative, the number of nonshadow pixels which are identified correctly
FP = 0;     % false positive, the number of nonshadow pixels which are identified as true shadow pixels
FN = 0;     % false negative, the number of true shadow pixels which are identified as nonshadow pixels

%% Statistics
% Compute the Accuracy metrics
sum = m*n;
for i = 1:m
    for j = 1:n 
        if result(i,j) == 1 && GT(i,j) == 1
            TP = TP +1;
        elseif result(i,j) == 0 && GT(i,j) == 0
            TN = TN +1;
        elseif result(i,j) == 1 && GT(i,j) == 0
            FP = FP +1;
        elseif result(i,j) == 0 && GT(i,j) == 1
            FN = FN +1;
        end
    end
end
% 
% %% Accuracy indexes
% % For BER index, the lower its value, the better the detection result is. 
% % Other indexes, the higher, the better.
% 
% % Producer's accuracies
% pro_s = double(TP)/double(TP+FN);
% pro_n = double(TN)/double(FP+TN);
% 
% % User's accuracies
% user_s = double(TP)/double(TP+FP);
% user_n = double(TN)/double(TN+FN);
% 
% % Overall accuracy
% Total = double(TP+TN)/double(sum);
% 
% % F-score accuracy
% F = (2*pro_s*user_s)/(pro_s+user_s);
% 
% 
% % Balance Error Rate (BER)
% BER = 1-(pro_s+pro_n)/2;   
% 
% % Output as a matrix
% % acc = zeros(7,2);
% % acc(7,1) = ('pro_s','pro_n','user_s','user_n','Total','F-score','BER');
% %acc = cat(1,pro_s, pro_n, user_s, user_n, Total, F, BER);
% %acc = zeros(1,7);
% acc = cat(2,pro_s, pro_n, user_s, user_n, Total, F, BER);



