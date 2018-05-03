clear all;
clc;

p_W_corners = uiimport('../data/p_W_corners.txt');
p_W_corners = p_W_corners.p_W_corners;

detected_corners = uiimport('../data/detected_corners.txt');
detected_corners = detected_corners.detected_corners;

K = uiimport('../data/K.txt');
K = K.K;

srcFiles = dir('../data/images_undistorted/*.jpg');
p_W_corners = p_W_corners/100;
%% 
detR = [];
quats = [];
trans = [];
for i = 1:length(srcFiles)
    
      p = detected_corners(i,:);
      [Rcw, tcw] = estimatePoseDLT(p, p_W_corners, K);
      M = [Rcw, tcw];
      [p_reprojected] = reprojectPoints(p_W_corners, M, K);
      Rwc = inv(Rcw);
      twc = -inv(Rcw)*tcw;
      qwc = rotMatrix2Quat(Rwc);
      quats = [quats qwc];
      trans = [trans twc];
      filename = strcat(srcFiles(i).folder, '\', srcFiles(i).name);
      image = imread(filename);
      figure(1);
      imshow(image);
      hold on;
      plot(p_reprojected(:,1), p_reprojected(:,2),'+');
      hold on;
      p = reshape(p, 2, 12)';
      plot(p(:,1), p(:,2),'+');
      hold off;
      pause(0.01);
end

%plotTrajectory3D(30, trans, quats, p_W_corners);