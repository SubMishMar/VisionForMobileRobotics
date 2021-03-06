function [R_C_W, t_C_W, query_keypoints, all_matches, best_inlier_mask, ...
    max_num_inliers_history] = ransacLocalization(...
    query_image, database_image, database_keypoints, p_W_landmarks, K)
% query_keypoints should be 2x1000
% all_matches should be 1x1000 and correspond to the output from the
%   matchDescriptors() function from exercise 3.
% best_inlier_mask should be 1xnum_matched (!!!) and contain, only for the
%   matched keypoints (!!!), 0 if the match is an outlier, 1 otherwise.

harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 1000;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 5;

% Find Harris scores of the query image
harris_scores = harris(query_image, harris_patch_size, harris_kappa);
assert(min(size(harris_scores) == size(query_image)));

% Find Keypoints of the query image
query_keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius);

% Find Descriptors of the query image
query_descriptors = describeKeypoints(query_image, query_keypoints, descriptor_radius);

% Find Descriptors of the database image
database_descriptors = describeKeypoints(database_image, database_keypoints, descriptor_radius);

% Matches
all_matches = matchDescriptors(query_descriptors, database_descriptors, match_lambda);

[~, query_indices, match_indices] = find(all_matches);

query_matches = flipud(query_keypoints(:, query_indices));
matched_landmarks = p_W_landmarks(:, match_indices);



use_DLT = false;

if use_DLT
    % Initialize RANSAC.
    best_inlier_mask = zeros(1, size(query_matches, 2));
    num_iterations = 2000;
    pixel_tolerance = 10;
    k = 6;

    max_num_inliers_history = zeros(1, num_iterations);
    max_num_inliers = 0;
    min_inlier_count = 6;    
    for i = 1:num_iterations
 
        [landmark_sample, idx] = datasample(matched_landmarks, k, 2, 'Replace', false);
        keypoint_sample = query_matches(:, idx);
        [R, t] = estimatePoseDLT(keypoint_sample, landmark_sample, K);
    
        M = [R, t];
    
        p_reprojected = reprojectPoints(matched_landmarks, M, K);
        errors = sum((query_matches - p_reprojected).^2, 1);
        is_inlier = errors < pixel_tolerance^2;
    
        if nnz(is_inlier) > max_num_inliers && ...
                nnz(is_inlier) >= min_inlier_count
                max_num_inliers = nnz(is_inlier);        
                best_inlier_mask = is_inlier;
        end
    
        max_num_inliers_history(i) = max_num_inliers;
    
    end

    [R_C_W, t_C_W] = estimatePoseDLT(query_matches(:, best_inlier_mask>0), matched_landmarks(:, best_inlier_mask>0), K);
else
    % Initialize RANSAC.
    best_inlier_mask = zeros(1, size(query_matches, 2));
    num_iterations = 200;
    pixel_tolerance = 10;
    k = 3;
    
    max_num_inliers_history = zeros(1, num_iterations);
    max_num_inliers = 0;
    min_inlier_count = 6;    
    
    for i = 1:num_iterations
        [landmark_sample, idx] = datasample(matched_landmarks, k, 2, 'Replace', false);
        keypoint_sample = query_matches(:, idx);
        normalized_bearings = K\[keypoint_sample; ones(1, 3)];
        for ii = 1:3
            normalized_bearings(:,ii) = normalized_bearings(:,ii)/norm(normalized_bearings(:,ii), 2);
        end
        poses = p3p(landmark_sample, normalized_bearings);
        R_C_W = zeros(3, 3, 2);
        t_C_W = zeros(3, 1, 2);
        for ii = 0:1
            R_W_C_ii = real(poses(:, (2+ii*4):(4+ii*4)));
            t_W_C_ii = real(poses(:, (1+ii*4)));
            R_C_W(:,:,ii+1) = R_W_C_ii';
            t_C_W(:,:,ii+1) = -R_W_C_ii'*t_W_C_ii;
        end
        
        M = [R_C_W(:,:,1), t_C_W(:,:,1)];
        p_reprojected = reprojectPoints(matched_landmarks, M, K);
        errors = sum((query_matches - p_reprojected).^2, 1);
        is_inlier = errors < pixel_tolerance^2;        
        
        M = [R_C_W(:,:,2), t_C_W(:,:,2)];
        p_reprojected = reprojectPoints(matched_landmarks, M, K);
        errors = sum((query_matches - p_reprojected).^2, 1);
        alternative_is_inlier = errors < pixel_tolerance^2;
        if nnz(alternative_is_inlier) > nnz(is_inlier)
            is_inlier = alternative_is_inlier;
        end
        if nnz(is_inlier) > max_num_inliers && ...
                nnz(is_inlier) >= min_inlier_count
                max_num_inliers = nnz(is_inlier);        
                best_inlier_mask = is_inlier;
        end
    
        max_num_inliers_history(i) = max_num_inliers;
    end
   [R_C_W, t_C_W] = estimatePoseDLT(query_matches(:, best_inlier_mask>0), matched_landmarks(:, best_inlier_mask>0), K);

end

end









