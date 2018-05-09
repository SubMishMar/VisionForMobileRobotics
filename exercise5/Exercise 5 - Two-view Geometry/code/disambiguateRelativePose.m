function [R,T] = disambiguateRelativePose(Rots,u3,points0_h,points1_h,K0,K1)
% DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
% four possible configurations) by returning the one that yields points
% lying in front of the image plane (with positive depth).
%
% Arguments:
%   Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
%   u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
%   p1   -  3xN homogeneous coordinates of point correspondences in image 1
%   p2   -  3xN homogeneous coordinates of point correspondences in image 2
%   K1   -  3x3 calibration matrix for camera 1
%   K2   -  3x3 calibration matrix for camera 2
%
% Returns:
%   R -  3x3 the correct rotation matrix
%   T -  3x1 the correct translation vector
%
%   where [R|t] = T_C1_C0 = T_C1_W is a transformation that maps points
%   from the world coordinate system (identical to the coordinate system of camera 0)
%   to camera 1.
%

R0 = Rots(:,:,1);
R1 = Rots(:,:,2);

M1 = K0*[eye(3,3), zeros(3,1)];

M2 = K1*[R0, u3];
P1 = linearTriangulation(points0_h,points1_h,M1,M2);
nP1_z = numel(find(P1(3,:)>0));

M2 = K1*[R0, -u3];
P2 = linearTriangulation(points0_h,points1_h,M1,M2);
nP2_z = numel(find(P2(3,:)>0));

M2 = K1*[R1, u3];
P3 = linearTriangulation(points0_h,points1_h,M1,M2);
nP3_z = numel(find(P3(3,:)>0));

M2 = K1*[R1, -u3];
P4 = linearTriangulation(points0_h,points1_h,M1,M2);
nP4_z = numel(find(P4(3,:)>0));

nP = [nP1_z, nP2_z, nP3_z, nP4_z];
[~, I] = max(nP);

if I == 1
    R = R0; T = u3;
elseif I == 2
    R = R0; T = -u3;
elseif I == 3
    R = R1; T = u3;
else
    R = R1; T = -u3;
end

end

