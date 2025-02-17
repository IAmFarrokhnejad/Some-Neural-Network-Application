%This MATLAB script is for training and evaluating an MLP feedforward neural network using the Crab dataset.
% 1: Data Loading, Network Initialization, Visualization, Data Splitting, Training Parameters Setting, Network Training, Performance Evaluation

[X,T] = crab_dataset;

net=newff(X, T, 20);
view(net)

net.layers{1}.transferFcn = 'logsig';
view(net) 

Xtr=X(:,1:170);
Ttr=T(:,1:170);
Xts=X(:,171:200);
Tts=T(:,171:200);

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


% 2: This section is similar to Section 1 but creates a neural network with two hidden layers, each containing 20 neurons. The training and evaluation process remains the same.

net=newff(X, T, [20, 20]);
view(net)

net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
view(net) 

net.divideParam.trainratio=1.0;
net.divideParam.valratio=0.0;
net.divideParam.testratio=0.0;
net.trainParam.goal=1e-7;

net=train(net,Xtr,Ttr);

Yts=net(Xts);
[c,cm] = confusion(Tts,Yts)


% 3: Target Data Modification, Network Initialization, Data Splitting, raining Parameters Setting, Network Training, Performance Evaluation

T=0.8*T+0.1;


net=newff(X, T, 20, {'logsig'});
view(net)

Xtr=X(:,1:170);
Ttr=T(:,1:170);
Xts=X(:,171:200);
Tts=T(:,171:200);

net.divideParam.trainratio=1.0;
net.divideParam.valratio=0.0;
net.divideParam.testratio=0.0;
net.trainParam.goal=1e-7;

net=train(net,Xtr,Ttr);

Yts=net(Xts);
[c,cm] = confusion(Tts,Yts)

Yts=net(Xts)
Tts

vec2ind(Yts)

[vec2ind(Yts)' vec2ind(Tts)']

sum((vec2ind(Yts)-vec2ind(Tts))>0)