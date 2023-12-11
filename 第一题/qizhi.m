clc;clear;close
%信号起止位置估计
M = 64; % 帧长
z = 32; % 帧移
symbol_rate = 10e4; % 发送符号间隔
bits_per_symbol = 1; % 每符号比特数
snr = 10;
sig_per = 0.5;

window = hamming(16);   % 窗口长度
overlap = 8;   % 窗口重叠长度
fs = 40e6;

[Bpsk_Signal,noise] = Bpsk_Signal_generater(symbol_rate,bits_per_symbol,snr,sig_per);
I = (length(Bpsk_Signal) - M)/z - 1; % 二维矩阵的行数

frames = buffer(Bpsk_Signal, M, M-z, 'nodelay');% 信号分帧
frames = frames';   
noise_frames = buffer(noise, M, M-z, 'nodelay');% 噪声分帧
noise_frames = noise_frames';

numFrames = size(frames, 1);% 获取信号帧数

P_start_noise1 = 0;counter = 1;
for i = 1:numFrames
    [Pxx_1, ~] = pwelch(frames(i,:), window, overlap, [], fs);
    P1(i,:) = Pxx_1;
    [Pxx_2, ~] = pwelch(noise_frames(i,:), window, overlap, [], fs);
    P2(i,:) = Pxx_2;
    jiance(i,:) = Pxx_1 - Pxx_2; %检查信号存在的数据段
    if jiance(i,:) ~= 0
        jiance1(counter) = i;
        counter = counter + 1;
    end
    if i <= 5
        P_start_noise1 = P2(i,:) + P_start_noise1;
    end
end
fprintf("信号存在位置:");
fprintf("%d ",jiance1);

%取信号的前5帧噪声的平均值作为噪声的功率谱估计值
P_start_noise = P_start_noise1/5;

%计算信号与噪声的功率谱距离熵
counter = 1;
for i = 1:numFrames
    distanceMatrix(i,:) = abs(sqrt(P1(i,:)) - sqrt(P_start_noise)); % 使用欧几里得距离计算距离
    % 每一帧信号的功率谱归一化
    % distanceMatrix_normalized = distanceMatrix(i,:) / sum(distanceMatrix(i,:));
    % 每一帧信号的功率谱距离熵
    % power_spectrum_entropy(i) = -sum(distanceMatrix_normalized .* log2(distanceMatrix_normalized));
    power_spectrum_entropy(i) = -sum(distanceMatrix(i,:).^2 .* log(distanceMatrix(i,:).^2)); % 功率谱距离熵
    if i <= 5
        distanceMatrix_noise(i,:) = abs(sqrt(P2(i,:)) - sqrt(P_start_noise)); % 使用欧几里得距离计算距离
        % 每一帧信号的功率谱归一化
        % distanceMatrix_normalized = distanceMatrix_noise(i,:) / sum(distanceMatrix_noise(i,:));
        % 每一帧信号的功率谱距离熵
        % power_spectrum_entropy_noise = -sum(distanceMatrix_normalized .* log2(distanceMatrix_normalized));
        power_spectrum_entropy_noise = -sum(distanceMatrix_noise(i,:).^2 .* log(distanceMatrix_noise(i,:).^2)); % 功率谱距离熵
        H_star_noise(i) = power_spectrum_entropy_noise;
    end
    if i > numFrames-5
        distanceMatrix_noise(i,:) = abs(sqrt(P2(i,:)) - sqrt(P_start_noise)); % 使用欧几里得距离计算距离
        % 每一帧信号的功率谱归一化
        % distanceMatrix_normalized = distanceMatrix_noise(i,:) / sum(distanceMatrix_noise(i,:));
        % % 每一帧信号的功率谱距离熵
        % power_spectrum_entropy_noise = -sum(distanceMatrix_normalized .* log2(distanceMatrix_normalized));
        power_spectrum_entropy_noise = -sum(distanceMatrix_noise(i,:).^2 .* log(distanceMatrix_noise(i,:).^2)); % 功率谱距离熵
        H_end_noise(counter) = power_spectrum_entropy_noise;
        counter = counter + 1;
    end
end

%开始5帧噪声信号的功率谱距离熵均值
start_noise_entropy_avg = sum(H_star_noise)/5;
%末尾5帧噪声信号的功率谱距离熵均值
end_noise_entropy_avg = sum(H_end_noise)/5;
%阈值设置
if start_noise_entropy_avg > end_noise_entropy_avg
    H_noise = 1*max(H_star_noise) + 0*start_noise_entropy_avg;
else
    H_noise = 1*max(H_end_noise) + 0*end_noise_entropy_avg;
end

counter = 1;
for i = 1:numFrames
    if power_spectrum_entropy(i) > H_noise
        mark(counter) = i;
        counter = counter + 1;
    end
end

first_mark = mark(1);
last_mark = mark(counter-1);

fprintf("\n标记位置:");
fprintf("%d ",mark);

fprintf("\n起始帧:%d\n",first_mark);
fprintf("最后帧:%d",last_mark);

Singal_start = frames(first_mark,:);
Singal_start_first = Singal_start(z+1);
Singal_start_last = Singal_start(M);

Singal_end = frames(last_mark,:);
Singal_end_start = Singal_end(1);
Singal_end_last = Singal_end(z-1);

fprintf("\n起始信号存在范围:");
fprintf("%d —— %d",Singal_start_first,Singal_start_last);
fprintf("\n结束信号存在范围:");
fprintf("%d —— %d\n",Singal_end_start,Singal_end_last);

