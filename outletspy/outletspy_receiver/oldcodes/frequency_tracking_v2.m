% 可修改参数
searching_frequency = 66800;
searching_frequency_width = 30;
% 路径
indir = 'testingset_1111.wav';
fs = 192000;
wav_hours = 0;
wav_minutes = 48;
wav_seconds = 18;
wav_len = wav_hours * 60 * 60 + wav_minutes * 60 + wav_seconds;
window_len = 1;
window_overlap = 0.5;
window_number = floor((wav_len - window_len) / (window_len - window_overlap));
results = zeros(window_number, 2);
tic;
for ii = 1:window_number
    start_time = ii * (window_len - window_overlap);
    samples = [round(start_time * fs + 1), round((start_time + window_len) * fs)];
    data = audioread(indir, samples);
    data = data(:,2);
    N = length(data);
    x_fft = abs(fft(data));
    x_f = 0 : fs / N : fs - fs / N;
    searching_start = searching_frequency - searching_frequency_width;
    searching_end = searching_frequency + searching_frequency_width;
%     x_search_index_high = find(x_f > searching_start);
%     x_search_index_low = find(x_f < searching_end);
    x_search_index = round(searching_start * N / fs) : round(searching_end * N / fs);
    x_fft_for_search = x_fft(x_search_index);
    x_f_for_search = x_f(x_search_index);
    [M, I] = max(x_fft_for_search);
    pfc_frequency = x_f_for_search(I);
    searching_frequency = pfc_frequency;
    results(ii, :) = [start_time, pfc_frequency]; 
    if mod(ii, 100) == 0
        fprintf('%d out of %d is completed!\n', ii, window_number);
    end
end
toc;
% filter_window_size = 10;
% filter_window_function = ones(1, filter_window_size) / filter_window_size;
% results_for_filter = [ones(10, 1) * results(1,2);results(:,2)];
% results_filtered = filter(filter_window_function, 1, results_for_filter);
% results(:,2) = results_filtered(11:end);
csvwrite("frequency.csv", results);
figure
plot(results(:,1), results(:,2));