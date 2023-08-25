clear

address = 'V0_bb_850/T850_1000_20220909_141416_613_00000_channel0.tiff';
% address = 'data/T1700_data3_norm/T1700_channel_548.tiff';
data = imread(address);
info = imfinfo(address);

x = 

plot(data[80, :], );