function calclas(W, x, N_N)

% Calculate the classes

    P=x/norm(x);
    
	for j=1:N_N,
	  ii(j)=sum((P-W(:,j)').^2);
	end;

	[m n]=min(ii);

    disp([' Class  of the entered vector : ', num2str(n)]);