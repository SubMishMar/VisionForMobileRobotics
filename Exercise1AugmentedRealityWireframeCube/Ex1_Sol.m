clc
clear
addpath(genpath('data'))
poses = load('data/poses.txt');
D = load('data/D.txt');
K = load('data/K.txt');
no_of_pics = 736;
cube_corners = [ 4, 0,  0;
                 4, 4,  0;
                 8, 4,  0;
                 8, 0,  0;
                 8, 0, -4;
                 4, 0, -4;
                 4, 4, -4;
                 8, 4, -4];
for i = 1:no_of_pics
      
      omega = poses(i,1:3);
      t = poses(i,4:6);
      t = t(:);
      k = omega./norm(omega);
      theta = norm(omega);
      k_skew = [ 0    -k(3)   k(2);
                 k(3)    0   -k(1);
                -k(2)  k(1)   0];
      R = eye(3,3) + sin(theta)*k_skew + (1 - cos(theta))*k_skew*k_skew;
      X = 0:4:32;
      Y = 0:4:20;
      UV = zeros(2, length(X)*length(Y));
      l = 1;
      for j = 1:length(X)
          for k = 1:length(Y)
              pc = [R t]*[X(j)/100;Y(k)/100;0;1];
              x = pc(1)/pc(3);
              y = pc(2)/pc(3);
%               r = sqrt(x^2 + y^2);
%               x = (1 + D(1)*r^2 + D(2)*r^4)*x;
%               y = (1 + D(1)*r^2 + D(2)*r^4)*y;
              UV(:,l)  = K(1:2,1:3)*[x;y;1];
              l = l + 1;
          end
      end
      
    if i<= 9
       Idistorted = imread(strcat('img_000',num2str(i),'.jpg'));
    elseif i > 9 && i <= 99 
       Idistorted = imread(strcat('img_00',num2str(i),'.jpg'));
    else
        Idistorted = imread(strcat('img_0',num2str(i),'.jpg'));
    end 
    I = undistort_image(Idistorted,K,D);
    imshow(I)
    hold on;
    plot(UV(1,:),UV(2,:),'ro',...
    'LineWidth',2,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[1 0 0],...
    'MarkerSize',10);
    hold on;
    m = 1;
    UV_cube = zeros(2,length(cube_corners));
    for j = 1:length(cube_corners)
        XYZ = cube_corners(j,:);
        pc = [R t]*[XYZ(1)/100;XYZ(2)/100;XYZ(3)/100;1];
        x = pc(1)/pc(3);
        y = pc(2)/pc(3);
%         r = sqrt(x^2 + y^2);
%         x = (1 + D(1)*r^2 + D(2)*r^4)*x;
%         y = (1 + D(1)*r^2 + D(2)*r^4)*y;
        UV_cube(:,m)  = K(1:2,1:3)*[x;y;1];
        m = m + 1;
    end
    
    plot(linspace(UV_cube(1,1),UV_cube(1,2),100),...
         linspace(UV_cube(2,1),UV_cube(2,2),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,2),UV_cube(1,3),100),...
         linspace(UV_cube(2,2),UV_cube(2,3),100),...
         'r','linewidth',3); 
    hold on;
    plot(linspace(UV_cube(1,3),UV_cube(1,4),100),...
         linspace(UV_cube(2,3),UV_cube(2,4),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,4),UV_cube(1,1),100),...
         linspace(UV_cube(2,4),UV_cube(2,1),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,1),UV_cube(1,6),100),...
         linspace(UV_cube(2,1),UV_cube(2,6),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,6),UV_cube(1,5),100),...
         linspace(UV_cube(2,6),UV_cube(2,5),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,5),UV_cube(1,4),100),...
         linspace(UV_cube(2,5),UV_cube(2,4),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,5),UV_cube(1,4),100),...
         linspace(UV_cube(2,5),UV_cube(2,4),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,6),UV_cube(1,7),100),...
         linspace(UV_cube(2,6),UV_cube(2,7),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,7),UV_cube(1,2),100),...
         linspace(UV_cube(2,7),UV_cube(2,2),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,7),UV_cube(1,8),100),...
         linspace(UV_cube(2,7),UV_cube(2,8),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,8),UV_cube(1,3),100),...
         linspace(UV_cube(2,8),UV_cube(2,3),100),...
         'r','linewidth',3);
    hold on;
    plot(linspace(UV_cube(1,8),UV_cube(1,5),100),...
         linspace(UV_cube(2,8),UV_cube(2,5),100),...
         'r','linewidth',3);
    hold on;
    pause(0.01)
    hold off;
end

function I = undistort_image(Idistorted,K,D)
addpath(genpath('data'));
Idistorted = rgb2gray(Idistorted);
Idistorted = im2double(Idistorted);

fx = K(1,1);
fy = K(2,2);
cx = K(1,3);
cy = K(2,3);
k1 = D(1);
k2 = D(2);
k3 = 0;
p1 = 0;
p2 = 0;

I = zeros(size(Idistorted));
[i, j] = find(~isnan(I));

% Xp = the xyz vals of points on the z plane
Xp = K\[j i ones(length(i),1)]';

% Now we calculate how those points distort i.e forward map them through the distortion
r2 = Xp(1,:).^2+Xp(2,:).^2;
x = Xp(1,:);
y = Xp(2,:);

x = x.*(1+k1*r2 + k2*r2.^2) + 2*p1.*x.*y + p2*(r2 + 2*x.^2);
y = y.*(1+k1*r2 + k2*r2.^2) + 2*p2.*x.*y + p1*(r2 + 2*y.^2);

% u and v are now the distorted cooridnates
u = reshape(fx*x + cx,size(I));
v = reshape(fy*y + cy,size(I));

% Now we perform a backward mapping in order to undistort the warped image coordinates
I = interp2(Idistorted, u, v);
end
