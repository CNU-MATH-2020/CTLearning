function rec = medfuncBackProjection(theta_num, N, R1, delta)
% 直接反投影重建
rec = zeros(N); % 重建后的像素值
for m = 1: theta_num
    pm = R1(: , m); % 取某一角度的投影数据
    Cm = (N / 2) * (1 - cos((m - 1) * delta) - sin((m - 1) * delta));
    for k1 = 1: N
        for k2 = 1: N
            Xrm = Cm + (k2 - 1) * cos((m - 1) * delta) + (k1 - 1) * sin((m - 1) * delta);
            n = floor(Xrm); % 射束编号
            t = Xrm - floor(Xrm); % 小数部分
            n = max(1, n);n = min(n, N-1); % 限定n的范围为1~N-1
            p = (1 - t) * pm(n) + t * pm(n + 1); % 线性内插
            rec(N + 1 - k1, k2) = rec(N + 1 - k1, k2) + p; % 反投影，图像需要翻转90°
        end
    end
end

