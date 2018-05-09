function [newpts, T] = normalise2dpts(pts)
% NORMALISE2DPTS - normalises 2D homogeneous points
%
% Function translates and normalises a set of 2D homogeneous points 
% so that their centroid is at the origin and their mean distance from 
% the origin is sqrt(2).
%
% Usage:   [newpts, T] = normalise2dpts(pts)
%
% Argument:
%   pts -  3xN array of 2D homogeneous coordinates
%
% Returns:
%   newpts -  3xN array of transformed 2D homogeneous coordinates.
%   T      -  The 3x3 transformation matrix, newpts = T*pts
%

pts_mean = mean(pts, 2);
pts_mean_removed = pts - pts_mean;

N = size(pts, 2);
sum = 0;
for i = 1:N
    sum = sum + pts_mean_removed(1,i)^2 + pts_mean_removed(2,i)^2;
end
sum = sum/N;
sum = sqrt(sum);

pts_sigma = sum;

s = sqrt(2)/pts_sigma;

u_x = pts_mean(1);
u_y = pts_mean(2);

T = [s 0 -s*u_x;
     0 s -s*u_y;
     0 0  1];
 
newpts = T*pts;




