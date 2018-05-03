function [p_reprojected] = reprojectPoints(P, M, K)
 P = [P ones(length(P), 1)];
 P = P';
 p= (K*M*P)';
 p_reprojected = [];
 for i = 1:length(p)
     p_reprojected = [p_reprojected; p(i,1)/p(i,3), p(i,2)/p(i,3)];
 end
end