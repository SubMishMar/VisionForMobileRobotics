function M = estimatePoseDLT(p, P, K)
% size(p) = [12, 2]
% size(P) = [12, 3]
 
 p_clbtd_cord = (K\ ([p ones(12,1)])')';
 Q = [];
 for i = 1:12
     X = P(i, 1); Y = P(i, 2); Z = P(i, 3);
     x = p_clbtd_cord(i,1); y =  p_clbtd_cord(i, 2);
     Q = [Q;
          X Y Z 1 0 0 0 0 -x*X -x*Y -x*Z -x;
          0 0 0 0 X Y Z 1 -y*X -y*Y -y*Z -y];
 end
 [~,~,V] = svd(Q);
 M = reshape(V(:,end),4,3)';
 if M(3,4) < 0
         M = -M;
 end
 R = M(:,1:3);
 [U,~,V] = svd(R);
 R_tld = U*V';
 alpha = norm(R_tld,'fro')/norm(R,'fro');
 M = [R_tld alpha*M(:,4)];
end