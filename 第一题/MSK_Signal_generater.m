function MSK_Signal = MSK_Signal_generater(symbol_rate, bits_per_symbol, snr_dB, signal_percentage)
    num_symbols = floor(256 * signal_percentage);

    % Generate random symbol indices (0 or 1 for MSK)
    data = randi([0, 1], 1, num_symbols);

    % Generate MSK modulated signal
    t_signal = 0:1/symbol_rate:(num_symbols * bits_per_symbol / symbol_rate - 1/symbol_rate);
    msk_signal = cos(2 * pi * symbol_rate * t_signal + pi/2 * cumsum(data));

    % Add zero sequences at the front and back
    signal_front = floor(rand * (256 - num_symbols));
    signal_back = 256 - num_symbols - signal_front;
    msk_signal_combine = [zeros(1, signal_front), msk_signal, zeros(1, signal_back)];

    % Upsample and filter
    upsampled_signal = upsample(msk_signal_combine, 4);
    rolloff = 0.35; % Rolloff factor
    span = 10; % Filter span in symbol durations
    sps = 4; % Samples per symbol
    rcosine_filter = rcosdesign(rolloff, span, sps);
    filtered_signal = filter(rcosine_filter, 1, upsampled_signal);

    % Add noise
    noise_power = 10^(-snr_dB/10);
    noise = sqrt(noise_power/2) * (randn(1, length(filtered_signal)) + 1i * randn(1, length(filtered_signal)));
    received_signal = filtered_signal + noise;

    MSK_Signal = received_signal;

    % Plotting
    figure;
    subplot(3, 1, 1);
    plot(t_signal, msk_signal);
    title('MSK Modulated Signal');

    subplot(3, 1, 2);
    plot(t_signal, received_signal);
    title('Received Signal');

    subplot(3, 1, 3);
    t_received = 0:1/symbol_rate:(length(received_signal)/symbol_rate - 1/symbol_rate);
    plot(t_received, real(received_signal), 'r');
    hold on;
    plot(t_received, imag(received_signal), 'b');
    title('Received Signal (Real and Imaginary)');
    legend('Real', 'Imaginary');
end
