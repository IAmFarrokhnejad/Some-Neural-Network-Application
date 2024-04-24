% Training and testing using the Wine dataset. Both sections essentially perform similar tasks: they load the Wine dataset, initialize neural networks, train them, and evaluate their performance using confusion matrices. However, they employ different methods for creating and training the neural networks.



% 1: Data Preparation, Neural Network Initialization, Data Division, Training Configuration, Neural Network Training, Testing

[x,t] = wine_dataset;

t=0.8*t+0.1;

rinx=randperm(178);
xt=x(:,rinx);
tt=t(:,rinx);

net=newff(xt, tt, 20);
view(net)

net.layers{1}.transferFcn = 'logsig';

Xtr=xt(:,1:150);
Ttr=tt(:,1:150);
Xts=xt(:,151:178);
Tts=tt(:,151:178);

net.divideParam.trainratio=1.0;
net.divideParam.valratio=0.0;
net.divideParam.testratio=0.0;
net.trainParam.goal=1e-7;

net=train(net,Xtr,Ttr);

Ytr=net(Xtr);
plotconfusion(Ttr,Ytr)

Yts=net(Xts);
plotconfusion(Tts,Yts)

[c,cm] = confusion(Tts,Yts)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));


% 2: Data Preparation, Neural Network Initialization, Network Configuration, Data Division, Training, Testing

[x,t] = wine_dataset;
net = patternnet(10);
view(net)

net=configure(net, x, t);
view(net)

net.divideParam.trainratio=0.80;
net.divideParam.valratio=0.00;
net.divideParam.testratio=0.20;

[net tr]=train(net,x,t);

xt=x(:, tr.testInd);
tt=t(:,tr.testInd);


xt=x(:, tr.testInd);
tt=t(:,tr.testInd);
yt=net(xt);
plotconfusion(tt,yt)

[c,cm] = confusion(tt,yt)