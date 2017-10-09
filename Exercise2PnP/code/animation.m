clc
clear all
addpath(genpath('data'))
K = load('data/K.txt');
p_W_corners = 0.01*load('data/p_W_corners.txt');
detected_corners = load('data/detected_corners.txt');
transl = [];
quats = [];

for i=1:210

    p = reshape(detected_corners(i,:), 2, 12)';

    % Now that we have the 2D <-> 3D correspondences (pts2d+normalized <-> p_W_corners),
    % let's find the camera pose with respect to the world using DLT
    M = estimatePoseDLT(p, p_W_corners, K);
    
    R_C_W = M(1:3,1:3);
    t_C_W = M(1:3,4);
    rotMat = R_C_W';
    quats(i,:) = rotMatrix2Quat(rotMat);
    transl(i,:) = -R_C_W' * t_C_W;

end

%% Generate video of the camera motion

fps = 30;
plotTrajectory3D(fps, transl', quats', p_W_corners');
