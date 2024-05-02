function plotmap2(w, Dim, iter)
%PLOTMAP  Plots feature map diagram using any neighborhood function.
%
%         PLOTMAP2(W, Dim, Iter)
%           W  - RxS matrix of weight vectors.
%           Dim - row vector of neighborhood parameters.
%           Iter - Number of iterations.
%
%          Plots the feature map layer by connecting points
%          representing weight row vectors to points representing
%          direct neighbors.
%
%         Original Function is written by,
%         M.H. Beale & H.B. Demuth, 1-31-92
%         Copyright (c) 1992 by the MathWorks, Inc.
%
%                                           Modified by,
%                                           A. Rizaner, 2008

if nargin ~= 3
  error('Wrong number of arguments.');
  end

[wr,wc] = size(w);
if wc < 2
  error('W must have at least two columns.');
  end

holdst = ishold;

% PLOT DOT AT FIRST NEURON
% (INCASE MAP IS TOO SMALL TO SEE)

plot(w(1,1),w(1,2),'.')

% CONNECT NEIGHBORS WITH LINES

clf
hold on

for i=1:wr
  
  n=ngbhr1(i, Dim);
  n = n(find(n > i));
  nl = length(n);
  
  for j=1:nl
  
    plot([w(i,1) w(n(j),1)],[w(i,2) w(n(j),2)]);
    plot([w(i,1) w(n(j),1)],[w(i,2) w(n(j),2)],'*');
    
  end
    
  end

% LABEL AXES
xlabel('W(i,1)');
ylabel('W(i,2)');
title(['Iterations = ', num2str(iter)]);

if ~holdst, hold off, end