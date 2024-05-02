function nb=neigh2d(Indx, Dim, Rds)

% NB=NEIGH2D(INDX, DIM, RDS)
%
% Two-dimensional Neighborhood Function
% calculates the neighbors of the winning unit
%
%  NB    :  Indexes of the neighbors
%  DIM   :  Dimension of the neruson e.g. 10x10
%  INDX  :  Index of the winning unit
%  RDS   :  Radius
%
%                       Ahmet Rizaner, 2008
%

Indx1=1:Dim(1)*Dim(2);
Indx2=[ceil(Indx/Dim(1)), Indx-(ceil(Indx/Dim(1))-1)*Dim(1) ];

Indxb=(Indx2-Rds);
Indxe=Indxb+(2*Rds);

if Indxb(1)<1, Indxb(1)=1; end;
if Indxb(2)<1, Indxb(2)=1; end;

if Indxe(1)>Dim(1), Indxe(1)=Dim(1); end;
if Indxe(2)>Dim(2), Indxe(2)=Dim(2); end;

Matx=vec2mat(Indx1,Dim(1),Dim(2));

IndxA=Matx(Indxb(1):Indxe(1), Indxb(2):Indxe(2));

DimA=prod(size(IndxA));

nb=reshape(IndxA',1,DimA);