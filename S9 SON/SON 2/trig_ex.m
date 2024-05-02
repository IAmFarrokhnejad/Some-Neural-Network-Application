%   ITEC 460 - Neural Computations
%   SOFM - Triangle
%	

clear;

% INITIALIZING THE PARAMATAERS

h=2;                    % Height of the triangle
w=1;                    % Half width of the triangle
no_iter=50000;         % Number of iteration needed
Disp_Rate_1=250;        % Display rate 1
Disp_Rate_2=5000;      % Display rate 2
N_I=2;                  % Number of input
N_N=65;                 % Number of neuron
N_A=1;                  % Neighbourhood area, radius
ETHA=0.9;               % Learning rate
figcnt=1;               % Figure Counter

% Initializing the weights
W=2*rand(N_I, N_N)-1;

A0=W;

% THE PROGRAM IS THERE

  for iter=1:no_iter,

	% Intput vector
    
    x2=2*rand;
	x1=(w-(w/h)*x2)*rand*sign(rand-rand);
	P=[ x1 x2 ];	% The input data


 % The learning rate ETHA is =>

	ETHA=1/(0.005*iter+1);

	if iter>10000, 
		ETHA=0.01; 
	end;

    
% The Neighbourhood functions width, radius

	if iter>22000,
        N_A=0;
    elseif iter>10000,
        N_A=1;
    elseif iter>5000,
		N_A=2;
    elseif iter>1000,
        N_A=3;
    else
        N_A=4;
	end;
    

 % Finding the minumum distance =>

	for j=1:N_N,
	  ii(j)=sum((P-W(:,j)').^2);
	end;

	[m n]=min(ii);

% Calculate the neighbors of the winning unit

	neigh=neigh1d(n,N_N,N_A);


    
 % Adjusting the weights of the winning neuron and its neighbourhoods


	for k=1:max(size(neigh)),

	  W(:,neigh(k))=W(:,neigh(k))+ETHA*(P'-W(:,neigh(k)));

	end;


    
% Ploting desicion boundaries
 figcnt=1;

if iter<5*Disp_Rate_1,
    
	if rem(iter,Disp_Rate_1)==0 || iter==1,
		        figure(figcnt);
                plot(W(1,:),W(2,:),W(1,:),W(2,:),'*b');
                xlabel('W(1,i)');
                ylabel('W(2,i)');
                title(['Iteration = ', num2str(iter)]);
                figcnt=figcnt+1;
	end;
    
else,
    
    	if rem(iter,Disp_Rate_2)==0,
		        figure(figcnt);
                plot(W(1,:),W(2,:),W(1,:),W(2,:),'*b');
                xlabel('W(1,i)');
                ylabel('W(2,i)');
                title(['Iteration = ', num2str(iter)]);
                figcnt=figcnt+1;
        end;
    
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