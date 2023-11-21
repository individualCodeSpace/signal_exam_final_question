%%读取数据并计算功率谱，功率谱熵%%
% 加载 MAT 文件
loaded_data = load('BPSK_signal_data.mat');

% 获取结构体数组
data_array = loaded_data.data_array;

% 设置功率谱计算的参数
fs = 40e6; % 采样率为40M
window = hamming(512); % 汉明窗口，窗长为512
noverlap = 256; % 重叠长度设置为窗长的一半
power_sum = [];
% 设置功率谱熵判决门限,查表知漏检概率0.1时，标准正态判决门限为1.28
for i = 1:numel(data_array)
    Bpsk_Signal = data_array(i).data;
    if data_array(i).sig_per == 0
        [Pxx_1, f] = pwelch(Bpsk_Signal, window, noverlap, [], fs);
        Pxx_normalized_1 = Pxx_1 / sum(Pxx_1);
        power_spectrum_entropy_1 = -sum(Pxx_normalized_1 .* log2(Pxx_normalized_1));
        power_sum = [power_sum, power_spectrum_entropy_1];
    end 
end
H_noise = power_sum / numel(data_array(1).snr);

% 初始化预测准确数和错误数
right_count = 0;
err_count = 0;

% 遍历结构体数组并计算功率谱和功率谱熵
for i = 1:numel(data_array)
    rec_Bpsk_Signal = data_array(i).data;
    
    % 计算功率谱
    [Pxx, f] = pwelch(rec_Bpsk_Signal, window, noverlap, [], fs);

    % 计算功率谱熵
    Pxx_normalized = Pxx / sum(Pxx);
    power_spectrum_entropy = -sum(Pxx_normalized .* log2(Pxx_normalized));
    
    % 依据功率谱熵判决门限判断是否存在信号
    if power_spectrum_entropy < H_noise
        right_count = right_count + 1;
    else 
        err_count = err_count + 1;   
    end
    
    
    
    %{
    % 绘制功率谱图
    figure;
    subplot(2, 1, 1);
    plot(f, 10*log10(Pxx));
    plot(f, Pxx);
    title(['Power Spectral Density - SNR: ', num2str(data_array(i).snr), ', Signal Percentage: ', num2str(data_array(i).sig_per)]);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');

    % 显示功率谱熵
    subplot(2, 1, 2);
    bar(power_spectrum_entropy);
    title(['Power Spectral Entropy - SNR: ', num2str(data_array(i).snr), ', Signal Percentage: ', num2str(data_array(i).sig_per)]);
    xlabel('Entropy');
    ylabel('Value');
    %}
end
% 性能评估，预测准确度与信噪比的关系，预测准确度与占空比（信号长度）的关系
acc = (err_count / 11);
disp(acc)