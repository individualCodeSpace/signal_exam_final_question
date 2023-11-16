function Bpsk_Signal = Bpsk_Signal_generater(symbol_rate,bits_per_symbol,snr_dB,signal_percentage)

num_symbols = floor(256*signal_percentage); % 要发送的符号数量，在四倍采样速率情况下生成1024个采样点只需要256个符号即可
% 生成随机比特序列
data = randi([0 1], 1, num_symbols * bits_per_symbol);

% BPSK调制
symbols = 2 * data - 1;

% 生成时钟信号，基带信号符号的时间步长肯定是波特率的倒数。
t_signal = 0:1/symbol_rate:(num_symbols*bits_per_symbol/symbol_rate - 1/symbol_rate);

% 生成调制信号，
bpsk_signal = symbols .* cos(2 * pi * symbol_rate * t_signal);

% 加上无信号片段，为0序列
signal_front = floor(rand * (256 - num_symbols));
signal_back = 256 - num_symbols - signal_front;
bpsk_signal_combine = [zeros(1,signal_front),bpsk_signal,zeros(1,signal_back)];

% 添加高斯噪声
noise_power = 10^(-snr_dB/10);
noise = sqrt(noise_power/2) * (randn(1, length(bpsk_signal_combine)) + 1i * randn(1, length(bpsk_signal_combine)));
received_signal = bpsk_signal_combine + noise;
Bpsk_Signal = received_signal;
t = 0:1/symbol_rate:(length(bpsk_signal_combine)/symbol_rate - 1/symbol_rate);
% 显示发送信号和接收信号
figure;

subplot(3,1,1);
plot(t, bpsk_signal_combine);
title('BPSK发送信号');

subplot(3,1,2);
plot(t, received_signal);
title('BPSK接收信号');

subplot(3,1,3);
plot(t, real(received_signal), 'r');
hold on;
plot(t, imag(received_signal), 'b');
title('接收信号（红色：实部，蓝色：虚部）');
legend('实部', '虚部');
end