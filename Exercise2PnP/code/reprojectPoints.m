function p_reprojected = reprojectPoints(P, M, K)
    
    p_reprojected = [];
    for i = 1:12
        p_repro = K*M*[P(i,:)';1];
        u = p_repro(1)/p_repro(3);
        v = p_repro(2)/p_repro(3);
        p_reprojected = [p_reprojected u v];
    end
end