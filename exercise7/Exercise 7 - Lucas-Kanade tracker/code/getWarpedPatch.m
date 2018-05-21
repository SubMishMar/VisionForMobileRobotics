function patch = getWarpedPatch(I, W, x_T, r_T)
% x_T is 1x2 and contains [x_T y_T] as defined in the statement. patch is
% (2*r_T+1)x(2*r_T+1) and arranged consistently with the input image I.

% I_padded = padarray(I, [r_T, r_T]);
% x_T = x_T + r_T*ones(size(x_T));


patch = zeros(2*r_T+1);

n_rows = size(I, 1);
n_cols = size(I, 2);

for X = -r_T:r_T
    for Y = -r_T:r_T
        
        p_warped = x_T' + W*[X; Y; 1];
        
        x_warped = p_warped(1);
        y_warped = p_warped(2);
        
        x = floor(x_warped); 
        y = floor(y_warped);

        a = x_warped - x; b = y_warped - y;
        
        if x+1 > 0 && x+1 <= n_cols  && y+1 > 0 && y+1 <= n_rows
            
          patch(Y + r_T + 1, X + r_T + 1) = (1-b) * ((1-a)*I(y,x) + a*I(y,x+1)) ...
                                      + b * ((1-a)*I(y+1,x) + a*I(y+1,x+1));
        end
        
    end
end
        
end






