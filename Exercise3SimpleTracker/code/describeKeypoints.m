function descriptors = describeKeypoints(img, keypoints, r)
% Returns a (2r+1)^2xN matrix of image patch vectors based on image
% img and a 2xN matrix containing the keypoint coordinates.
% r is the patch "radius".
tempimg = padarray(img, [r, r]);
keypoints = keypoints + r*ones(size(keypoints));
descriptors = zeros((2*r+1)^2,1);
for i=1:length(keypoints)
    cord = keypoints(:,i);
    I = cord(1);
    J = cord(2);
    IntMat = tempimg(I-r:I+r,J-r:J+r);
    IntMat = reshape(IntMat,[(2*r+1)^2,1]);
    descriptors(:,i) = IntMat;
end

end
