%%% 将仿真信号存储起来 %%%
% 创建结构体数组
data_array_bpsk = struct('snr', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'sig_per', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'data', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)));
data_array_fsk4 = struct('snr', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'sig_per', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'data', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)));
data_array_msk = struct('snr', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'sig_per', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'data', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)));
data_array_cpsk8 = struct('snr', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'sig_per', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'data', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)));
data_array_qam16 = struct('snr', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'sig_per', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)), 'data', cell(1, numel(-20:2:10)*numel(0:0.1:0.9)));



% 计数器
counter = 1;

for snr = -20:2:10
    for sig_per = 0:0.1:0.9
        rec_Bpsk_Signal = Bpsk_Signal_generater(10e6,1,snr,sig_per);
        data_array_bpsk(counter).snr = snr;
        data_array_bpsk(counter).sig_per = sig_per;
        data_array_bpsk(counter).data = rec_Bpsk_Signal;
        counter = counter + 1;
    end
end
counter = 1;
for snr = -20:2:10
    for sig_per = 0:0.1:0.9
        rec_Bpsk_Signal = FSK4_Signal_generater(10e6,1,snr,sig_per);
        data_array_fsk4(counter).snr = snr;
        data_array_fsk4(counter).sig_per = sig_per;
        data_array_fsk4(counter).data = rec_Bpsk_Signal;
        counter = counter + 1;
    end
end
counter = 1;
for snr = -20:2:10
    for sig_per = 0:0.1:0.9
        rec_Bpsk_Signal = MSK_Signal_generater(10e6,1,snr,sig_per);
        data_array_msk(counter).snr = snr;
        data_array_msk(counter).sig_per = sig_per;
        data_array_msk(counter).data = rec_Bpsk_Signal;
        counter = counter + 1;
    end
end
counter = 1;
for snr = -20:2:10
    for sig_per = 0:0.1:0.9
        rec_Bpsk_Signal = CPSK8_Signal_generater(10e6,1,snr,sig_per);
        data_array_cpsk8(counter).snr = snr;
        data_array_cpsk8(counter).sig_per = sig_per;
        data_array_cpsk8(counter).data = rec_Bpsk_Signal;
        counter = counter + 1;
    end
end
counter = 1;
for snr = -20:2:10
    for sig_per = 0:0.1:0.9
        rec_Bpsk_Signal = QAM16_Signal_generater(10e6,1,snr,sig_per);
        data_array_qam16(counter).snr = snr;
        data_array_qam16(counter).sig_per = sig_per;
        data_array_qam16(counter).data = rec_Bpsk_Signal;
        counter = counter + 1;
    end
end
% 将结构体数组保存到文件
save('BPSK_signal_data.mat', 'data_array_bpsk');
save('FSK4_signal_data.mat', 'data_array_fsk4');
save('MSK_signal_data.mat', 'data_array_msk');
save('CPSK8_signal_data.mat', 'data_array_cpsk8');
save('QAM16_signal_data.mat', 'data_array_qam16');

