function F = fundamentalEightPoint_normalized(p1, p2)
% estimateEssentialMatrix_normalized: estimates the essential matrix
% given matching point coordinates, and the camera calibration K
%
% Input: point correspondences
%  - p1(3,N): homogeneous coordinates of 2-D points in image 1
%  - p2(3,N): homogeneous coordinates of 2-D points in image 2
%
% Output:
%  - F(3,3) : fundamental matrix
%

[newp1, T1] = normalise2dpts(p1);
[newp2, T2] = normalise2dpts(p2);

newp1 = newp1/newp1(end);
newp2 = newp2/newp2(end);

F = fundamentalEightPoint(newp1,newp2);

F = T2'*F*T1;

