function F = fundamentalEightPoint(p1,p2)
% fundamentalEightPoint  The 8-point algorithm for the estimation of the fundamental matrix F
%
% The eight-point algorithm for the fundamental matrix with a posteriori
% enforcement of the singularity constraint (det(F)=0).
% Does not include data normalization.
%
% Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.
%
% Input: point correspondences
%  - p1(3,N): homogeneous coordinates of 2-D points in image 1
%  - p2(3,N): homogeneous coordinates of 2-D points in image 2
%
% Output:
%  - F(3,3) : fundamental matrix

N = size(p1, 2);
Q = zeros(N, size(p1,1)*size(p2,1));
for i = 1:N
    p_1 = p1(:,i); 
    p_2 = p2(:,i);
    Q(i,:) = kron(p_1, p_2)';
end

[~, ~, V] = svd(Q);

F = reshape(V(:,end), 3, 3);
[U, S, V] = svd(F);
S(end,end) = 0;

F = U*S*V';

end