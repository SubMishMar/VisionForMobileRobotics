clc
clear
addpath(genpath('data'))
detected_corners = load('data/detected_corners.txt');
p_W_corners = 0.01*load('data/p_W_corners.txt');
K = load('data/K.txt');

i = 1;

p = detected_corners(i,:);
p = reshape(p, 2, 12)';

M = estimatePoseDLT(p, p_W_corners, K);
p_rprjtd = reprojectPoints(p_W_corners , M, K);
p_rprjtd = reshape(p_rprjtd, 2, 12)';
I = imread(strcat('img_000',num2str(1),'.jpg'));
imshow(I);
hold on;
p = plot(p(:,1),p(:,2),'bo');
hold on;
q = plot(p_rprjtd(:,1),p_rprjtd(:,2),'r+');
hold off;
legend('original points','reprojected points');
