function [W, p_hist] = trackKLT(I_R, I, x_T, r_T, num_iters)
% I_R: reference image, I: image to track point in, x_T: point to track,
% expressed as [x y]=[col row], r_T: radius of patch to track, num_iters:
% amount of iterations to run; W(2x3): final W estimate, p_hist 
% (6x(num_iters+1)): history of p estimates, including the initial
% (identity) estimate
