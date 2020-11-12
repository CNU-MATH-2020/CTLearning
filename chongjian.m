% 直接反投影重建 %
clc;
clear;
close all;

% 参数设置 %
N = 256;
I = phantom(N); % 生成S-L模型
theta = 0: 1: 179; % 投影角度
delta = pi / 180; % 角度增量
theta_num = length(theta);
d = 1; % 平移步长

% 产生投影数据 %
P = radon(I, theta);
[mm, nn] = size(P); % 投影的行、列的长度
e = floor((mm - N - 1) / 2 + 1) +1; % 投影数据的默认中心为floor((size(I)+1)/2)
P = P(e: N + e - 1, : ); % 截取中心N点数据
P1 = reshape(P, N, theta_num); 

% 直接反投影重建 %
rec = medfuncBackProjection(theta_num, N, P1, delta);

% R-L滤波 %
fh_RL = medfuncRlfilterfunction(N, d);

% R-L滤波反投影重建 %
rec_RL = medfuncRLfilteredbackprojection(theta_num, N, P1, delta, fh_RL);

% 投影结果显示 %
figure;
subplot(2, 2, 1), imshow(I, []), title('原始图像');
subplot(2, 2, 2), imshow(P1, []), title('截取后投影图像');
subplot(2, 2, 3), imshow(rec, []), title('直接反投影重建图像');
subplot(2, 2, 4), imshow(rec_RL, []), title('R-L滤波反投影重建图像');