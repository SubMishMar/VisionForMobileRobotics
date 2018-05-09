function E = estimateEssentialMatrix(p1, p2, K1, K2)
% estimateEssentialMatrix_normalized: estimates the essential matrix
% given matching point coordinates, and the camera calibration K
%
% Input: point correspondences
%  - p1(3,N): homogeneous coordinates of 2-D points in image 1
%  - p2(3,N): homogeneous coordinates of 2-D points in image 2
%  - K1(3,3): calibration matrix of camera 1
%  - K2(3,3): calibration matrix of camera 2
%
% Output:
%  - E(3,3) : fundamental matrix
%
[newp1, T1] = normalise2dpts(p1);
[newp2, T2] = normalise2dpts(p2);

newp1 = newp1/newp1(end);
newp2 = newp2/newp2(end);

F = fundamentalEightPoint_normalized(newp1,newp2);

F = T2'*F*T1;

E = K2'*F*K1;