function matches = matchDescriptors(...
    query_descriptors, database_descriptors, lambda)
% Returns a 1xQ matrix where the i-th coefficient is the index of the
% database descriptor which matches to the i-th query descriptor.
% The descriptor vectors are MxQ and MxD where M is the descriptor
% dimension and Q and D the amount of query and database descriptors
% respectively. matches(i) will be zero if there is no database descriptor
% with an SSD < lambda * min(SSD). No two non-zero elements of matches will
% be equal.

[D,I] = pdist2(database_descriptors',query_descriptors','euclidean','smallest',1);
dnonzeroMin = min(D(find(D~=0)));
I(find(D>=lambda*dnonzeroMin)) = 0;

uniquematches = zeros(size(I));
[~,uniquematchindexs,~] = unique(I, 'stable');
uniquematches(uniquematchindexs) = I(uniquematchindexs);

matches = uniquematches;



end
