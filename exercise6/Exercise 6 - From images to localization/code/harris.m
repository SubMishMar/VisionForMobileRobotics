function scores = harris(img, patch_size, kappa)
sobel_x = [-1 0 1;
           -2 0 2;
           -1 0 1];
sobel_y = sobel_x';
Ix = conv2(img, sobel_x, 'valid');
Iy = conv2(img, sobel_y, 'valid');
Ix2 = Ix.*Ix;
Iy2 = Iy.*Iy;
IxIy = Ix.*Iy;

patch = ones(patch_size)./(patch_size^2);

sumIx2 = conv2(Ix2, patch, 'valid');
sumIy2 = conv2(Iy2, patch, 'valid');
sumIxy = conv2(IxIy, patch, 'valid');

detMuv = sumIx2.*sumIy2 - sumIxy.*sumIxy;
trace2Muv = (sumIx2 + sumIy2).^2;

scores = detMuv - kappa*trace2Muv;
scores(find(scores < 0)) = 0;
scores = padarray(scores, [(patch_size+1)/2 (patch_size+1)/2]);
end
