function [best_guess_history, max_num_inliers_history] = ...
    parabolaRansac(data, max_noise)
% data is 2xN with the data points given column-wise, 
% best_guess_history is 3xnum_iterations with the polynome coefficients 
%   from polyfit of the BEST GUESS SO FAR at each iteration columnwise and
% max_num_inliers_history is 1xnum_iterations, with the inlier count of the
%   BEST GUESS SO FAR at each iteration.
N = 100;
best_guess_history = zeros(3, N);
max_num_inliers_history = zeros(1, N);
max_num_inliers = 0;
for count=1:N
 sampled_data = datasample(data, 3, 2, 'Replace',false);
 p = polyfit(sampled_data(1,:)', sampled_data(2,:)',2);
 y_m = para_model(p, data(1,:));
 
 compare_y = abs(data(2,:) - y_m)<=max_noise;
 inlier_idx = find(compare_y==1);
 inlier_count = numel(find(inlier_idx));
 
 if inlier_count > max_num_inliers
     max_num_inliers = inlier_count;
     best_guess = polyfit(data(1,inlier_idx), data(2, inlier_idx),2);
 end
max_num_inliers_history(count) = max_num_inliers;
best_guess_history(:,count) = best_guess(:)';
end
 

end
