function I = warpImage(I_R, W)

n_rows = size(I_R, 1);
n_cols = size(I_R, 2);

I = zeros(size(I_R));

for j = 1:n_cols
    for i = 1:n_rows
        p_new = floor(W*[j;i;1]);
        x_new = p_new(1);
        y_new = p_new(2);
        if x_new >=1 && y_new > 1 && x_new <= n_cols && y_new <= n_rows
            I(i, j) = I_R(y_new, x_new);
        end
    end
end


