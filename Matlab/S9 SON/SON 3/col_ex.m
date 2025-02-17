%   ITEC 460 - Neural Computations
%   SOFM - Color Seperation	

clear;

% INITIALIZING THE PARAMATAERS


no_iter=5000;         % Number of iteration needed
Disp_Rate_1=100;        % Display rate 1
Disp_Rate_2=1000;      % Display rate 2
N_I=3;                  % Number of input
N_N=3;                 % Number of neuron
N_A=1;                  % Neighbourhood area, radius
ETHA=0.9;               % Learning rate
figcnt=1;               % Figure Counter

% Initializing the weights
W=2*rand(N_I, N_N)-1;

A0=W;

% Datas:
xi(1,:)=[255, 0, 0];
xi(2,:)=[0, 255, 0];
xi(3,:)=[0, 0, 255];
xi(4,:)=[139, 123, 191];
xi(5,:)=[44, 50, 170];
xi(6,:)=[191, 95, 23];
xi(7,:)=[202, 121, 12];
xi(8,:)=[20, 194, 148];
xi(9,:)=[84, 144, 70];

% THE PROGRAM IS THERE

  for iter=1:no_iter,

	% Intput vector
    
    a=floor(rand*9)+1;
	P=xi(a,:)/norm(xi(a,:));
    

 % The learning rate ETHA is =>

	ETHA=1/(0.01*iter+1);

	if iter>100, 
		ETHA=0.01; 
	end;

    
% The Neighbourhood functions width, radius

	if iter>500,
        N_A=0;
    elseif iter>100,
        N_A=1;
    elseif iter>50,
		N_A=2;
    elseif iter>10,
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
                plot(W(1,:),W(2,:),W(1,:),W(2,:),'sq b');
                xlabel('W(1,i)');
                ylabel('W(2,i)');
                title(['Iteration = ', num2str(iter)]);
                figcnt=figcnt+1;
	end;
    
else,
    
    	if rem(iter,Disp_Rate_2)==0,
		        figure(figcnt);
                plot(W(1,:),W(2,:),W(1,:),W(2,:),'sq b');
                xlabel('W(1,i)');
                ylabel('W(2,i)');
                title(['Iteration = ', num2str(iter)]);
                figcnt=figcnt+1;
        end;
    
end;    



end;



% Calculate the classes

for i=1:9,
    P=xi(i,:)/norm(xi(i,:));
    
	for j=1:N_N,
	  ii(j)=sum((P-W(:,j)').^2);
	end;

	[m n]=min(ii);

    disp(['Input Vector : ', num2str(i), ' Class : ', num2str(n)]);

end;