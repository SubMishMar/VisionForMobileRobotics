function [points, intensities] = disparityToPointCloud(...
    disp_img, K, baseline, left_img)
% points should be 3xN and intensities 1xN, where N is the amount of pixels
% which have a valid disparity. I.e., only return points and intensities
% for pixels of left_img which have a valid disparity estimate! The i-th
% intensity should correspond to the i-th point.

b_ = [baseline;0;0];
n_rows = size(disp_img, 1);
n_cols = size(disp_img, 2);
P = [];
intensities = [];

for x = 1:n_cols
    for y = 1:n_rows
        if disp_img(y, x) > 0
        p0 = [x; y; 1];
        p1 = p0 - [disp_img(y, x); 0; 0];
        A = inv(K)*[p0 -p1];
        lambda = pinv(A)*b_;
        P = [P lambda(1)*inv(K)*p0];
        intensities = [intensities, left_img(y,x)];
        end
    end
end

points = P;

