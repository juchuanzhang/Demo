clc;clear;
tic;
filter_frequency = csvread("frequency.csv");
% ·��
indir = 'testingset_1111.wav';

filter_width = 25;
Fs=192000;
wav_hours = 0;
wav_minutes = 48;
wav_seconds = 18;
total_len = wav_hours * 60 * 60 + wav_minutes * 60 + wav_seconds;
window_len = 4;
window_num = floor(total_len / window_len);
Fs_downsample = 192;
% results = zeros(window_len * Fs_downsample * window_num, 1);
results_raw = zeros(window_len * Fs, window_num);

parfor ii = 1:window_num 
    current_time = (ii - 1) * window_len;
    start_ind = current_time * Fs + 1;
    end_ind = start_ind + window_len * Fs - 1;
    samples = [start_ind, end_ind];
    data=audioread(indir, samples);
    data_1=data(:,2);
    N=length(data_1);        
    t_end1 = length(data_1)/Fs; % ��Ƶ��ʱ��
    t1 = 0:1/Fs:t_end1 - 1/Fs; % ʱ��������    
    [~, I] = min(abs(current_time - filter_frequency(:,1)));
    center_frequency = filter_frequency(I, 2);
    passband_low = center_frequency - filter_width;
    passband_high = center_frequency + filter_width;     
    ts = timeseries(data_1, t1);
    ts_filtered = idealfilter(ts, [passband_low, passband_high], 'pass');
    x_ifft = ts_filtered.Data;
    results_raw(:,ii) = x_ifft; 
    fprintf("No. %d is OK!\n", ii);
end
%%
results_raw = reshape(results_raw, [window_len * Fs * window_num, 1]);
   % �벨ȡƽ��
results_raw = max(results_raw, 0); % ȡ�벨
% ��ͨ�˲�
t = (0 : 1 / Fs : (length(results_raw) - 1) / Fs)';
ts = timeseries(results_raw, t);
ts_filtered = idealfilter(ts, [0, 25], 'pass');
x_ifft_filtered = ts_filtered.Data;
results = resample(x_ifft_filtered, 1, 1000);   

%%
figure;
t_all = (0 : 1 / Fs_downsample : (length(results) - 1) / Fs_downsample)';
csvwrite("output.csv", [t_all, results])
toc;
plot(t_all, results);
