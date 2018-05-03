function disp_img = getDisparity(...
    left_img, right_img, r, min_disp, max_disp)
% left_img and right_img are both H x W and you should return a H x W
% matrix containing the disparity d for each pixel of left_img. Set
% disp_img to 0 for pixels where the SSD and/or d is not defined, and for d
% estimates rejected in Part 2. patch_radius specifies the SSD patch and
% each valid d should satisfy min_disp <= d <= max_disp.



disp_img = zeros(size(left_img));

nrows = size(left_img, 1);
ncols = size(left_img, 2);

I0 = padarray(left_img, [r, r]);
I1 = padarray(right_img, [r, r]);

disparity = min_disp:max_disp;
patch1_vecs = zeros((2*r+1)^2, size(disparity,2));

for i = 1+r:nrows-r
    for j = r+max_disp+1:ncols-r
        p0 = [i+r, j+r];
        row = p0(1);
        col = p0(2);
        patch0 = I0(row-r:row+r, col-r:col+r);
        patch0 = patch0(:);
        count = 1;
        for d=min_disp:max_disp
            p1 = p0 - [0, d];
            row = p1(1);
            col = p1(2);
            patch1 = I1(row-r:row+r, col-r:col+r);
            patch1 = patch1(:);
            patch1_vecs(:,count) = patch1;
            count = count+1;
        end
        [D, I] = pdist2(single(patch1_vecs'), single(patch0'), 'euclidean','smallest', length(disparity));
        number = numel(find(D(2:end)<=1.5*min(D)));
        disp = disparity(I(1));

        if disp ~= min_disp && disp ~= max_disp && number <= 2 
           

           disp_left = disp-1;
           disp_right = disp+1;

           indx1 = find(I == find(disparity == disp_left));
           indx2 = find(I == find(disparity == disp_right));

           x = [disp_left, disp, disp_right];
           y = [D(indx1),D(1),D(indx2)];
           p = polyfit(x(:)', y(:)', 2);

           d_min = -p(2)/(2*p(1));
           disp_img(i, j) = d_min;
        end
    end
end


