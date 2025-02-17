% Demontration of the usage of the "newrb" function to create and train a radial basis network for function approximation

net = newrb(P,T,goal,spread,MN,DF);
P = -1:.1:1;
T = [-.9602 -.5770 -.0729 .3771 .6405 .6600 .4609 .1336 -.2013 -.4344 -.5000 -.3930 -.1647 .0988 .3072 .3960 .3449 .1816 -.0312 -.2189 -.3201];
plot(P,T,'+');
title('Training Vectors');
xlabel('Input Vector P');
ylabel('Target Vector T');

x = -3:.1:3;
a = radbas(x);
plot(x,a)
title('Radial Basis Transfer Function');
xlabel('Input p');
ylabel('Output a');
a2 = radbas(x-1.5);
a3 = radbas(x+2);
plot(x,a,'r',x,a2,'b',x,a3,’g’)
grid

% When the spread of the radial basis neurons is too low
eg = 0.02; % sum-squared error goal
sc = .01; % spread constant
net = newrb(P,T,eg,sc);
NEWRB, neurons = 0, MSE = 0.176192
plot(P,T,'+');
X = -1:.01:1;
Y = net(X);
hold on;
plot(X,Y);
hold off;

% When the spread of the radial basis neurons is too high

eg = 0.02; % sum-squared error goal
sc = 100; % spread constant
net = newrb(P,T,eg,sc);
NEWRB, neurons = 0, MSE = 0.176192
plot(P,T,'+');
X = -1:.01:1;
Y = net(X);
hold on;
plot(X,Y);
hold off;


eg = 0.02; % sum-squared error goal
sc = 1; % spread constant
net = newrb(P,T,eg,sc);
NEWRB, neurons = 0, MSE = 0.176192
plot(P,T,'+');
X = -1:.01:1;
Y = net(X);
hold on;
plot(X,Y);
hold off;