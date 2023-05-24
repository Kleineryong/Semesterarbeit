clear
addpath(genpath(pwd))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = {'data/T1897_0_digital/digital_value_1897.xlsx',
 'data/T1897_2_digital/digital_value_1897.xlsx',
 'data/T1897_3_digital/digital_value_1897.xlsx'};
final_data = cell(size(filename));
for i = 1:numel(filename)
    [~, sheetnames] = xlsfinfo(filename{i});
    for j = 1:numel(sheetnames)
        data(:, :, j) = xlsread(filename{i}, sheetnames{j});
    end
    final_data{i} = data; 
end

train_data = [reshape(final_data{1}, 2500, 9); reshape(final_data{2}, 2500, 9); reshape(final_data{3}, 2500, 9)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_t_name = {'data/T1897_0_digital/t_field_1897.xlsx',
 'data/T1897_2_digital/t_field_1897.xlsx',
 'data/T1897_3_digital/t_field_1897.xlsx'};
final_t_data = cell(size(file_t_name));
for i = 1:numel(file_t_name)
    [~, sheet_t_names] = xlsfinfo(file_t_name{i});
    for j = 1:numel(sheet_t_names)
        data_t(:, :, j) = xlsread(file_t_name{i});
    end
    final_t_data{i} = data_t; 
end
train_t_data = [reshape(final_t_data{1}, 2500, 1); reshape(final_t_data{2}, 2500, 1); reshape(final_t_data{3}, 2500, 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_emi_name = {'data/T1897_0_digital/emi_field_1897.xlsx',
 'data/T1897_2_digital/emi_field_1897.xlsx',
 'data/T1897_3_digital/emi_field_1897.xlsx'};
final_emi_data = cell(size(file_emi_name));
for i = 1:numel(file_emi_name)
    [~, sheet_emi_names] = xlsfinfo(file_emi_name{i});
    for j = 1:numel(sheet_emi_names)
        data_emi(:, :, j) = xlsread(file_emi_name{i});
    end
    final_emi_data{i} = data_emi; 
end
train_emi_data = [reshape(final_emi_data{1}, 2500, 1); reshape(final_emi_data{2}, 2500, 1); reshape(final_emi_data{3}, 2500, 1)];

train_target = [train_t_data, train_emi_data];


