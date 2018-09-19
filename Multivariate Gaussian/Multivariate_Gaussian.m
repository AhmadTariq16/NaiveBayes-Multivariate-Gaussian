clear all;
clc;

load 'banknote-traindata'
load 'banknote-testdata'


trainDataLabels=traindata(:,5);
testDataLabels=testdata(:,5);

train=traindata(:,1:4);
test=testdata(:,1:4);

[rowsTrain,colsTrain]=size(train);
[rowsTest,colTest]=size(test);

classes=unique(trainDataLabels);
numClasses=length(classes);

ytrain=(zeros(size(classes)));
for j=1:numClasses
    for i = 1:length(train())
        if(train(i,4)==classes(j))
        ytrain(j,1)=ytrain(j,1)+1;
        end
    end    
end

ytest=(zeros(size(classes)));
for j=1:numClasses
    for i = 1:length(test())
        if(test(i,4)==classes(j))
        ytest(j,1)=ytest(j,1)+1;
        end
    end    
end

class1Data=train((trainDataLabels==classes(1)),:); %training data belonging to class 0
class2Data=train((trainDataLabels==classes(2)),:); %training data belonging to class 1


% Class Probability P(Y=yk) for each class
for i=1:length(classes)
    numOfTrainDataLabels(i,:)= sum(trainDataLabels==classes(i));
end

classProb=zeros(size(classes));
for i=1:length(classes)
    classProb(i,:) = numOfTrainDataLabels(i)/length(trainDataLabels);
end


%Mean Estimates all the features given their classes
for i=1:numClasses
            tempDataTrain=train((trainDataLabels==classes(i)),:);
            meanTrain(i,:)=mean(tempDataTrain,1);
            
end

%Covariance Estimates
mean_subtract0 = bsxfun(@minus, class1Data, meanTrain(1,:));
cov0 = (mean_subtract0.' * mean_subtract0) / size(class1Data,1);

mean_subtract1 = bsxfun(@minus, class2Data, meanTrain(2,:));        
cov1 = (mean_subtract1.' * mean_subtract1) / size(class2Data,1);
cov={cov0, cov1};

p(:,1)=mvnpdf(test, meanTrain(1,:),cov0);
p(:,2)=mvnpdf(test, meanTrain(2,:),cov1);

for k=1:1:2
    
    Pr(:,k)=classProb(1).*p(:,k);
end

[pv0,id]=max(Pr,[],2);
for i=1:length(id)
    ClassificationTestDataLabels(i,1)=classes(id(i));
end

for i=1:length(ClassificationTestDataLabels)
    result(i,1)=testDataLabels(i);
    result(i,2)=ClassificationTestDataLabels(i);
end

temp=0;
for i=1:length(testDataLabels)
    if(testDataLabels(i,1)==ClassificationTestDataLabels(i,1))
        temp=temp+1;
    end
end
Accuracy=(temp/length(testDataLabels))*100

fileId =fopen('Output.txt', 'wt');
fprintf(fileId,'%g\n',ClassificationTestDataLabels);
fclose(fileId);



