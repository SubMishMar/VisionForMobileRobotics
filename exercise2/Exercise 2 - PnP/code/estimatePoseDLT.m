function [R, t] = estimatePoseDLT(p, P, K)
   p = reshape(p, 2, 12)';
   p = [p ones(length(p), 1)]';
   p = (K\p)';

   if size(p, 1) == size(P, 1)
       Q = [];
   for i = 1:size(p, 1)
       Q = [Q;
            P(i,:) 1 zeros(1, 4), -(p(i, 1)/p(i, 3))*[P(i,:) 1];
            zeros(1, 4)  P(i,:) 1 -(p(i, 2)/p(i, 3))*[P(i,:) 1]];
   end
   else
       disp('p!=P, check code');
   end
   [~, ~, V] = svd(Q);
   M = V(:,end);
   M = reshape(M, 4, 3)';
   if M(3,4) < 0
       M = -M;
   end
   Rtilde = M(1:3, 1:3);
   [U, ~, V] = svd(Rtilde);
   R = U*V';
   
   alpha = norm(R, 'fro')/norm(Rtilde, 'fro');
   
   t = alpha*M(:,4);
end