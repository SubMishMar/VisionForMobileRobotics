
close all;

% query = descriptors_2;
% database = descriptors;
% D = pdist2(query,database);

[D, I] = pdist2(descriptors_2',descriptors', 'euclidean', 'smallest', 1);
 dmin = min(D);
 k = find(D<=4*dmin);
 matches = I(k);
