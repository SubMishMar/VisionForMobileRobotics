function [p_reprojected] = reprojectPoints(P, M, K)
 P = [P; ones(1, length(P))];

 p = (K*M*P);
 
 for i=1:length(p)
     p(1,i) = p(1,i)/p(3,i);
     p(2,i) = p(2,i)/p(3,i);
 end
 
 p_reprojected = p(1:2,:);
end