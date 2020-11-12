function fh_RL = medfuncRlfilterfunction(N, d)
% R-L滤波
fh_RL = zeros(1, N);
for k1 = 1: N
    fh_RL(k1) = -1 / (pi ^ 2 * ((k1 - N / 2 - 1) * d) ^ 2);
    if mod(k1 - N / 2 - 1, 2) == 0
        fh_RL(k1) = 0;
    end
end
fh_RL(N / 2 + 1) = 1 / (4 * d ^ 2);

