function [dx, ssds] = trackBruteForce(I_R, I, x_T, r_T, r_D)
% I_R: reference image, I: image to track point in, x_T: point to track,
% expressed as [x y]=[col row], r_T: radius of patch to track, r_D: radius
% of patch to search dx within; dx: translation that best explains where
% x_T is in image I, ssds: SSDs for all values of dx within the patch
% defined by center x_T and radius r_D.

patch = getWarpedPatch(I_R, getSimWarp(0, 0, 0, 1), x_T, r_T);

table = zeros((2*r_D + 1)^2, 3);

ssds = zeros(2*r_D + 1);

count = 1;

for p5 = -r_D:r_D
    for p6 = -r_D:r_D
        guess = getWarpedPatch(I, getSimWarp(p5, p6, 0, 1), x_T, r_T);
        ssd = sum(sum((patch - guess).^2));
        table(count, : ) = [p5, p6, ssd];
        count = count + 1;
        ssds(p5 + r_D + 1, p6 + r_D + 1) = ssd;
    end
end

[~, idx] = min(table(:,3));
dx = table(idx,:);

end