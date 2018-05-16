function keypoints = selectKeypoints(scores, num, r)
% Selects the num best scores as keypoints and performs non-maximum 
% supression of a (2r + 1)*(2r + 1) box around the current maximum.
   scores = padarray(scores, [r,r]);
   keypoints = zeros(2, num);
   for m = 1:num
       temp_scores = scores(:);
       [~, I] = max(temp_scores);
       [row, col] = ind2sub(size(scores), I);
       keypoints(:,m) = [row;col];
       scores(row-r:row+r, col-r:col+r) = zeros(2*r+1);
   end
   keypoints = keypoints - r*ones(size(keypoints));
end
