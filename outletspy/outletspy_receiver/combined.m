% init frequency
start_time = 0;
time_len = 1;
Fs = 192000;
samples=[round(start_time*Fs + 1), round((start_time + time_len) * Fs)];
[data,Fs]=audioread('temp.wav',samples);
data_1=data(:,2);
x_fft=fft(data_1);
real_fft = abs(x_fft);
real_fft = real_fft(1:length(x_fft)/2);
region_of_interest = real_fft(65000:68000);
[M, I] = max(region_of_interest);
searching_frequency = I + 65000; % initial_pfc_freq
% figure;
window_len = 16384;
%specgram(data_1,window_len,Fs,window_len,0);
searching_frequency_width = 30;
% tracking
indir = 'temp.wav';
wav_len = 12;
window_len = 1;
window_overlap = 0.5;
window_number = floor((wav_len - window_len) / (window_len - window_overlap));
results = zeros(window_number, 2);
tic;
for ii = 1:window_number
    start_time = ii * (window_len - window_overlap);
    samples = [round(start_time * Fs + 1), round((start_time + window_len) * Fs)];
    data = audioread(indir, samples);
    data = data(:,2);
    N = length(data);
    x_fft = abs(fft(data));
    x_f = 0 : Fs / N : Fs - Fs / N;
    searching_start = searching_frequency - searching_frequency_width;
    searching_end = searching_frequency + searching_frequency_width;
    x_search_index = round(searching_start * N / Fs) : round(searching_end * N / Fs);
    x_fft_for_search = x_fft(x_search_index);
    x_f_for_search = x_f(x_search_index);
    [M, I] = max(x_fft_for_search);
    pfc_frequency = x_f_for_search(I);
    searching_frequency = pfc_frequency;
    results(ii, :) = [start_time, pfc_frequency]; 
end
toc;
% demodulation
filter_frequency = results;
filter_width = 25;
total_len = 12;
window_len = 4;
window_num = floor(total_len / window_len);
Fs_downsample = 192;
results_raw = zeros(window_len * Fs, window_num);

for ii = 1:window_num 
    current_time = (ii - 1) * window_len;
    start_ind = current_time * Fs + 1;
    end_ind = start_ind + window_len * Fs - 1;
    samples = [start_ind, end_ind];
    data=audioread(indir, samples);
    data_1=data(:,2);
    N=length(data_1);        
    t_end1 = length(data_1)/Fs;
    t1 = 0:1/Fs:t_end1 - 1/Fs;   
    [~, I] = min(abs(current_time - filter_frequency(:,1)));
    center_frequency = filter_frequency(I, 2);
    passband_low = center_frequency - filter_width;
    passband_high = center_frequency + filter_width;     
    ts = timeseries(data_1, t1);
    ts_filtered = idealfilter(ts, [passband_low, passband_high], 'pass');
    x_ifft = ts_filtered.Data;
    results_raw(:,ii) = x_ifft; 
end
%%
results_raw = reshape(results_raw, [window_len * Fs * window_num, 1]);
results_raw = max(results_raw, 0); 
t = (0 : 1 / Fs : (length(results_raw) - 1) / Fs)';
ts = timeseries(results_raw, t);
ts_filtered = idealfilter(ts, [0, 25], 'pass');
x_ifft_filtered = ts_filtered.Data;
results = resample(x_ifft_filtered, 1, 1000);   

% figure;
t_all = (0 : 1 / Fs_downsample : (length(results) - 1) / Fs_downsample)';
% figure
% plot(t_all, results);
% starting point detect
left_ptr = 1+56;
right_ptr = 96+56;
integral = sum(results(left_ptr:right_ptr));
sandglass = 0;
start_time = [];
threshold = 3e-3;
while (1)
    if (right_ptr >= length(t_all)-1)
        break;
    end
    integral = integral - results(left_ptr);
    left_ptr = left_ptr + 1;
    right_ptr = right_ptr + 1;
    integral = integral + results(right_ptr);
    if (integral >= threshold && sandglass == 0)
        start_time = [start_time left_ptr-56];
        sandglass = 192 * 5;
    end
    if sandglass
        sandglass = sandglass- 1;
    end

end

%csvwrite("output.csv", [t_all, results])
toc;


for i=1:1:length(start_time)
    if (start_time(i)+192*4-1<=length(t_all))
        this_time = clock
        this_time = this_time(4:6)
        csvwrite("output_csv/"+num2str(start_time(i)/192)+"_"+num2str(this_time(1))+num2str(this_time(2))+num2str(this_time(3))+".csv",[results(start_time(i):start_time(i)+192*4-1)]);
        figure
        plot(t_all(start_time(i):start_time(i)+192*4-1)-start_time(i)/192, results(start_time(i):start_time(i)+192*4-1));
    end
end
pause(4)
exit()
