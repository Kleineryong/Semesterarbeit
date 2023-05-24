test_digital_name = {'data/T1900_3_digital/digital_value_1900.xlsx',};
final_digital_data = cell(size(test_digital_name));
for i = 1:numel(test_digital_name)
    [~, sheet_digital_names] = xlsfinfo(test_digital_name{i});
    for j = 1:numel(sheet_digital_names)
        data(:, :, j) = xlsread(test_digital_name{i}, sheet_digital_names{j});
    end
    final_digital_data{i} = data; 
end

test_data = [reshape(final_digital_data{1}, 2500, 9)];
t_esti = zeros(length(test_data), 1);
emi_esti = zeros(length(test_data), 1);
for i = 1:length(test_data)
    test_result = sim(results.Network, transpose(test_data(i, :)));
    t_esti(i) = test_result(2);
    emi_esti(i) = test_result(1);
end

t_esti = reshape(t_esti, [50, 50]);
emi_esti = reshape(emi_esti, [50, 50]);

writematrix(t_esti, 'T1900_3_temp_esti_NN.xlsx');
writematrix(emi_esti, 'T1900_3_emi_esti_NN.xlsx');
