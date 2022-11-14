app_start_time = 1600;
time_len = 60;
Fs = 192000;
% f_sample = [0, 500, 1000, 1500, 2000, 2500, 3000];
% f_center = [65340, 65510, 65610, 65680, 65740, 65740, 65770];
% f_sample = [0, 1000, 2000, 5000, 10000, 15000];
% f_center = [6.58, 6.595, 6.605, 6.62, 6.635, 6.635];
f_sample = [0, 2000, 4000, 6000, 7800];
f_center = [6.614, 6.61, 6.618, 6.623, 6.626] * 10000;
% center = 66370
% figure;
% plot(f_sample, f_center);
p = polyfit(f_sample, f_center, 2);
filter_width = 25;
samples=[round(app_start_time*Fs + 1), round((app_start_time + time_len) * Fs)];
%         samples=[(41+(j-1)*15+(i-1)*1510)*Fs,(41+(j)*15+(i-1)*1510)*Fs];
%samples=[1,90*Fs];
[data,Fs]=audioread('testingset_1111.wav',samples);
data_1=data(:,2);
N=length(data_1);
x_fft=fft(data_1);
x_f_real=(0:Fs/N:Fs-Fs/N);
t_end1 = length(data_1)/Fs; % 音频总时长
t1 = 0:1/Fs:t_end1 - 1/Fs; % 时间坐标轴
window_len = 16384;
figure;
specgram(data_1,window_len,Fs,window_len,0);
%%
p_center = 66930;
passband_low = p_center - 25;
passband_high = p_center + 25;
% current_time = app_start_time;
% passband_low = polyval(p, current_time) - filter_width;
% passband_high = polyval(p, current_time) + filter_width; 
x_cut_index_high = find(x_f_real < passband_low);
x_fft(x_cut_index_high) = 0;
x_cut_index_low = find(x_f_real > Fs - passband_low);
x_fft(x_cut_index_low) = 0;
x_cut_index_high = find(x_f_real < Fs - passband_high);
x_cut_index_low = find(x_f_real > passband_high);
x_cut_index = intersect(x_cut_index_high, x_cut_index_low);
x_fft(x_cut_index) = 0;
x_ifft = real(ifft(x_fft)); 
%%
figure;
plot(x_ifft);
%%
% 半波取平均
x_ifft = max(x_ifft, 0); % 取半波
point_num = 1000;
x_ifft_downsampled = zeros(1, length(x_ifft) / point_num);
for k = 1:length(x_ifft) / point_num
x_ifft_downsampled(k) = mean(x_ifft((k - 1) * point_num + 1 : k * point_num));
end
%%
figure;
plot(x_ifft_downsampled)