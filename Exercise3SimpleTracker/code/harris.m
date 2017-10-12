function scores = harris(img, patch_size, kappa)
sobel_x = [ -1  0  1;
            -2  0  2;
            -1  0  1];
sobel_y = sobel_x';

Ix = conv2(img, sobel_x,'valid');
Iy = conv2(img, sobel_y,'valid');
Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy = Ix.*Iy;
patch = ones(patch_size)./81;
Ix2sum = conv2(Ix2, patch,'valid');
Iy2sum = conv2(Iy2, patch,'valid');
Ixysum = conv2(Ixy, patch,'valid');
pr = floor(patch_size / 2);
scores = Ix2sum.*Iy2sum - Ixysum.^2 ...
         - kappa*(Ix2sum + Iy2sum).^2;
scores(find(scores < 0)) = 0;
scores = padarray(scores, [1+pr 1+pr]);
end
