%   ITEC 460 - Neural Computations
%   SOFM - Square - 2D
%	

clear;

% INITIALIZING THE PARAMATAERS

no_iter=50000;          % Number of iteration needed
pausetime = 0.0;
Disp_Rate=1000;         % Display rate
N_I=2;                  % Number of input
XX=10;                  % X dimension
YY=10;                  % Y dimension
N_N=XX*YY;              % Number of neuron
N_A=1;                  % Neighbourhood area
ETHA=0.9;               % Learning rate
figcnt=0;               % Figure counter

W=rand(N_N,N_I);	% Initializing the weights

A0=W;	

% THE PROGRAM IS THERE

for iter=1:no_iter,

	
	x1=rand;
	x2=rand;
	%x3=rand;

%	P=[ x1 x2 x3 ];	% The input data

	P=[ x1 x2];	% The input data

 % The learning rate ETHA is =>

	ETHA=1/(0.009*iter+1.1);

	if iter>6000, 
		ETHA=0.01; 
	end;

% The Neighbourhood functions width

	if iter>9000,
		
		N_A=0;

	 elseif iter>4000,

		N_A=1;

       	 elseif iter>1000,

		N_A=2;
	else  
		N_A=3;

	end;


 % Finding the minumum distance =>

	for j=1:N_N,

	  ii(j)=sum((P-W(j,:)).^2);

	end;

	[m n]=min(ii);

	neigh=neigh2d(n,[XX YY],N_A);


 % Adjusting the weights of the winning neuron and its neighbourhoods


	for k=1:max(size(neigh)),

	  W(neigh(k),:)=W(neigh(k),:)+ETHA*(P-W(neigh(k),:));

	end;


% Ploting desicion boundaries

	if rem(iter,Disp_Rate)==0,

        figcnt=figcnt+1;
  %     figure(figcnt);
  
		pause2(pausetime);
		plotmap2d(W, [XX YY], iter);
	
	end;


% Store some examples

    if iter==50,
		A1=W;
	elseif iter==1000,
		A2=W;
	elseif iter==5000,
		A3=W;
	elseif iter==10000,
		A4=W;
	elseif iter==20000,
		A5=W;
	elseif iter==50000,
		A6=W;
	end;

end;