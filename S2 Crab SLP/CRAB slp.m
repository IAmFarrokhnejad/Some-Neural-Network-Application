%This code performs several tasks related to training and testing a single layer perceptron neural network using data from the CSV file 'crabs.csv'



M = csvread('crabs.csv')
M = readtable('crabs.csv')
net=perceptron;
view(net)
net=newp(X,T);
Xtr=X(:,1:170);
Ttr=T(:,1:170);
Xts=X(:,171:200);
Tts=T(:,171:200);
net=train(net,Xtr,Ttr);
Ytr=net(Xtr)
plotconfusion(Ttr,Ytr)
Yts=net(Xts)
plotconfusion(Tts,Yts)
[c,cm] = confusion(Tts,Yts)
net.trainParam.epochs=2000;
net.trainParam.goal=2e-2;
net.trainParam.lr=0.01;
