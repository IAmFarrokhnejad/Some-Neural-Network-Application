function plotmap1D(W, iter)
%PLOTMAP  Plots feature map diagram using any neighborhood function.
%
%         PLOTMAP1d(W, Dim, Iter)
%           W  - RxS matrix of weight vectors.
%           Iter - Number of iterations.
%
%          Plots the feature map layer by connecting points
%          representing weight row vectors to points representing
%          direct neighbors.
%

%
%                                           by,
%                                           A. Rizaner, 2008

if nargin ~= 2
  error('Wrong number of arguments.');
  end

[wr,wc] = size(W);
if wc < 2
  error('W must have at least two columns.');
end
                clf;
                plot(W(1,:),W(2,:),W(1,:),W(2,:),'*b');
                xlabel('W(1,i)');
                ylabel('W(2,i)');
                title(['Iteration = ', num2str(iter)]);