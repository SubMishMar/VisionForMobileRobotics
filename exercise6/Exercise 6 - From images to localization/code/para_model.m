function m = para_model(abc, data)
abc = abc(:)';
data_structured = [data.^2;
                  data;
                  ones(size(data))];
 m = abc*data_structured;
end