%This code demonstrates the implementation of a simple, single layer perceptron neural network using MATLAB

i=[ 8, 10, 6, 5, 15, 2, 20, 2; 5, 6, 2, 2, 6, 4, 4, 3];
t=[1, 1, 0, 0, 1, 0, 1, 0];

net=newp(i,t);
view(net)

net=adapt(net, i(:,1), t(1));
net.IW{:} % Weights
net.b{:} % Bias

[net, y, e]=adapt(net, i, t); 
net=train(net, i, t);

clear net;
net=newp(i,t);
net.trainParam.showCommandLine=true;
net=train(net, i, t);

net.layers{i}.transferFcn = 'hardlim';
net.layers{i}.transferFcn = 'hardlims';

i=[ 8, 10, 6, 5, 15, 2, 20, 2; 5, 6, 2, 2, 6, 4, 4, 3];
t=[1, 1, -1, -1, 1, -1, 1, -1];

net.layers{1}.transferFcn = 'hardlims’;
view(net) 

net=train(net, i, t);

i=[ 8, 10, 6, 5, 15, 2, 20, 2; 5, 6, 2, 2, 6, 4, 4, 3];

net=newp(i,t);
view(net)

net=train(net, i, t);

plotpv(i,t)
plotpc(net.IW{1,1},net.b{1})
net.iw{1,1}
net.b{1} 
plotpv(i,t)
plotpc(net.IW{1,1}(1,:),net.b{1}(1,:))
