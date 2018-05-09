function P = linearTriangulation(p1,p2,M1,M2)
% LINEARTRIANGULATION  Linear Triangulation
%
% Input:
%  - p1(3,N): homogeneous coordinates of points in image 1
%  - p2(3,N): homogeneous coordinates of points in image 2
%  - M1(3,4): projection matrix corresponding to first image
%  - M2(3,4): projection matrix corresponding to second image
%
% Output:
%  - P(4,N): homogeneous coordinates of 3-D points


P = zeros(4, length(p1));
for i=1:length(p1)
    
    p_1 = p1(:,i);
    p_2 = p2(:,i);
    
    p1_ = skew_mat(p_1);
    p2_ = skew_mat(p_2);
    
    A = [p1_ * M1;
         p2_ * M2];
    [~,~,V] = svd(A);
    
    P(:,i) = V(:,end)/V(end, end);
end