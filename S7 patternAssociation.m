%Implementation of a simple neural network model for pattern association 
%Initialization, Weight Calculation, Pattern Recognition, Visualization, Execution

S=[1 1 1, 1 -1 -1, 1 1 1, -1 -1 1, 1 1 1];
P=[1 1 1, 1 -1 1, 1 1 1, 1 -1 -1, 1 -1 -1]; 
E=[1 1 1, 1 -1 -1, 1 1 1, 1 -1 -1, 1 1 1]; 
U=[1 -1 1, 1 -1 1, 1 -1 1, 1 -1 1, 1 1 1]; 
T=[1 1 1, -1 1 -1, -1 1 -1, -1 1 -1, -1 1 -1];

pchar(S, 5, 3, 1)
pchar(P, 5, 3, 1)
pchar(E, 5, 3, 1)
pchar(U, 5, 3, 1)
pchar(T, 5, 3, 1)

Ws=S'*S;
Wp=P'*P;
We=E'*E;
Wu=U'*U;
Wt=T'*T;
W=Ws+Wp+We+Wu+Wt;

pchar(sign(S*W), 5, 3)


pchar(sign(S*W), 5, 3)
pchar(sign(P*W), 5, 3)

W=Ws+Wp+We+Wu;
pchar(sign(S*W), 5, 3)

% verpat.m is defined in the below
function verpat( inpat , weight )
%verpat(inpat)
% plots the input pattern and network output together
figure(1);
subplot(1,2,1), pchar(inpat,5, 3, 1); title('input');
subplot(1,2,2), pchar(sign(inpat*weight), 5, 3, 1); title('output');
end
% end of verpat.m

verpat(T,W)
verpat(corrupt(T,10),W)