function nb=neigh1d(indx, tot, rds)

% NB=NEIGH1D(INDX, TOT, RDS)
%
% One-dimensional Neighborhood Function
% calculates the neighbors of the winning unit
%
%  NB    :  Indexes of the neighbors
%  TOT   :  Total number of neurons
%  INDX  :  Index of the winning unit
%  RDS   :  Radius
%
%                       Ahmet Rizaner, 2008
%

all_nb=(indx-rds):(indx+rds);

inx_nb= find((all_nb<=tot) & (all_nb>0));

nb=all_nb(inx_nb);