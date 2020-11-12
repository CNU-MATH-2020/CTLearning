function rec_RL = medfuncRLfilteredbackprojection(theta_num, N, R1, delta, fh_RL)
% R-L滤波反投影
rec_RL = zeros(N);
for m = 1 : theta_num
    pm = R1(: , m); % 某一角度的投影值
    pm_RL = conv(fh_RL, pm, 'same'); % 做卷积，same表示返回与fh_RL大小相同的卷积中心部分
    Cm = (N / 2) * (1 - cos((m - 1) * delta) - sin((m - 1) * delta));
    for k1 = 1: N
        for k2 = 1: N
            Xrm = Cm + (k2 - 1) * cos((m - 1) * delta) + (k1 - 1) * sin((m - 1) * delta);
            n = floor(Xrm);
            t = Xrm - floor(Xrm);
            n = max(1, n); n = min(n, N - 1);
            p_RL = (1 - t) * pm_RL(n) + t * pm_RL(n+1); % 线性内插
            rec_RL(N + 1 - k1, k2) = rec_RL(N + 1 - k1, k2) + p_RL; % 反投影
        end
    end
end
