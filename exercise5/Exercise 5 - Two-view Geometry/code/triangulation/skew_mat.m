function p_skew = skew_mat(p)

 x = p(1); y = p(2); z = p(3);
 p_skew = [0 -z  y;
           z  0 -x;
          -y  x  0];
end