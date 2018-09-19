clear all;
clc;

load 'parkinson-Training.mat'
load parkinson-Test.mat

trainData=parkinson1(:,1:22);
trainDataLabels=parkinson1(:,23);
[rows,cols]=size(trainData);


testData=parkinson(:,1:22);
testDataWithLabels=parkinson;
testDataLabels=parkinson(:,23);

classes=unique(trainDataLabels);
numOfClasses=size(classes);

numOfTrainDataLabels= zeros(size(classes));

% Class Probability P(Y=yk) for each class
for i=1:length(classes)
    numOfTrainDataLabels(i,:)= sum(trainDataLabels==classes(i));
end

classProb=zeros(size(classes));
for i=1:length(classes)
    classProb(i,:) = numOfTrainDataLabels(i)/length(trainDataLabels);
end


%Conditional Means and Variances of all the features given their classes
for i=1:numOfClasses
            tempDataTrain=trainData((trainDataLabels==classes(i)),:);
            meanTrain(i,:)=mean(tempDataTrain,1);
            varianceTrain(i,:)=std(tempDataTrain,1);
end

% tempdata=[];
% for i=1:numOfClasses
%     for j= 1:length(trainDataLabels)
%         if(trainDataLabels==classes(i,1))
%             tempdata=trainData(j,:);
%         end
%     end
%    % meanTrain(i,:)= mean(tempdata);
% end
% 
sortedDataTest=[];

for i=1:numOfClasses
    sortedDataTest=vertcat(sortedDataTest,testDataWithLabels((testDataLabels==classes(i)),:));
    %tempDataTest=horzcat(tempDataTest,testDataLabels(i);
    %testDataCat(i,:)=mean(tempDataTest,1);         
end

y=(zeros(size(classes)));

for j=1:numOfClasses
    for i = 1:length(sortedDataTest())
        if(sortedDataTest(i,23)==classes(j))
        y(j,1)=y(j,1)+1;
        end
    end
    
end
lengthy=length(y);

% probs=zeros(size(1,22));
% for i=1:y(1,1)
%     for j=1:cols
%         probs(i,j)=normpdf(sortedDataTest(i,j),meanTrain(1,j),varianceTrain(1,j));
%     end
% end
% 
% for i=1:y(2,1)
%     for j=1:cols
%         probs0(i,j)=(normpdf(sortedDataTest(i,j),meanTrain(2,j),varianceTrain(2,j)));
%     end
% end



probXGiven0=zeros(size(testData));
for i=1:length(testDataLabels)
    for j=1:cols
        probXGiven0(i,j)=normpdf(testData(i,j),meanTrain(1,j),varianceTrain(1,j));
    end
end

probXGiven1=zeros(size(testData));
for i=1:length(testDataLabels)
    for j=1:cols
        probXGiven1(i,j)=(normpdf(testData(i,j),meanTrain(2,j),varianceTrain(2,j)));
    end
end

probClass0=ones(length(testDataLabels),1);
for i=1:length(testDataLabels)
    for j=1:cols
        probClass0(i,1)=probClass0(i,1)*probXGiven0(i,j);
    end
end


probClass1=ones(length(testDataLabels),1);
for i=1:length(testDataLabels)
    for j=1:cols
        probClass1(i,1)=probClass1(i,1)*probXGiven1(i,j);
    end
end

probs=horzcat(probClass0,probClass1);

classLabelNew=zeros(size(testDataLabels));
for i=1:length(testDataLabels)
    if(probs(i,1)>probs(i,2))
        classLabelNew(i,1)=0;
    else classLabelNew(i,1)=1;
    end
    
end

temp=0;
for i=1:length(testDataLabels)
    if(testDataLabels(i,1)==classLabelNew(i,1))
        temp=temp+1;
    end
end

Accuracy=(temp/length(testDataLabels))*100;

fileId =fopen('Output.txt', 'wt');
fprintf(fileId,'%g\n',classLabelNew);
fclose(fileId);
