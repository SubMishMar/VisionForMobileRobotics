function keypoints = selectKeypoints(scores, num, r)
% Selects the num best scores as keypoints and performs non-maximum 
% supression of a (2r + 1)*(2r + 1) box around the current maximum.
tempscores = padarray(scores, [r, r]);
keypoints = zeros(2,num);
for i = 1:num
     tempmat = tempscores(:);
     [~,locs] = max(tempmat);
     [I, J] = ind2sub(size(tempscores),locs);
     tempscores(I-r:I+r,J-r:J+r) = zeros(2*r+1);
     keypoints(:,i) = [I-r,J-r]; 
end
end

