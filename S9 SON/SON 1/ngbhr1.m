function nb=ngbhr1(Indx, Dim)

% NB=NGBHR1(INDX, Dim)
%
% Two-dimensional Neighborhood Function
% calculates the neighbors of the winning unit
% when radius is 1
%
%  NB    :  Indexes of the neighbors
%  DIM   :  Dimension of the neruson e.g. 10x10
%  INDX  :  Index of the winning unit
%
%                       Ahmet Rizaner, 2008
%

Indx1=ceil(Indx/Dim(1));
Indx2=Indx-(ceil(Indx/Dim(1))-1)*Dim(1);

Indr1=[Indx-1, Indx+1];
Indr2=find(Indr1>((Indx1-1)*Dim(1)) & Indr1<=Indx1*Dim(2));
Indr=Indr1(Indr2);

Indc1=[Indx-Dim(2), Indx+Dim(2)];
Indc2=find(Indc1>0 & Indc1<=Dim(1)*Dim(2));
Indc=Indc1(Indc2);

nb=sort([Indr, Indx, Indc]);