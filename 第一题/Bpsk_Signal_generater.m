function [Bpsk_Signal,noise] = Bpsk_Signal_generater(symbol_rate,bits_per_symbol,snr_dB,signal_percentage)
clc;close;

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

% 上采样内插，过升余弦滤波，采样因子为4，提升信号采样率，以实现长度1024的信号
upsampled_signal = upsample(bpsk_signal_combine,4);

%%% 平方升余弦滤波器 %%%
rolloff = 0.35; % 滚降系数
span = 10; % 滤波器的符号周期数
sps = 4; % 每个符号的样本数
rcosine_filter = rcosdesign(rolloff, span, sps);

% 对上采样后的信号进行滤波，去除插入零值引入的高频分量
filtered_signal = filter(rcosine_filter, 1, upsampled_signal);

signal_power = mean(filtered_signal.^2);
% 添加高斯白噪声
% noise_power = 10^(-snr_dB/10);
noise_power = signal_power / (10^(snr_dB/10));
noise = sqrt(noise_power) * (randn(1, length(filtered_signal)) + 1i * randn(1, length(filtered_signal)));
Bpsk_Signal = filtered_signal + noise;

t = 0:1/symbol_rate:(length(Bpsk_Signal)/symbol_rate - 1/symbol_rate);

% figure;
% subplot(2,1,1);
% plot(t_signal,bpsk_signal);
% subplot(2,1,2);
% plot(t,filtered_signal);

end