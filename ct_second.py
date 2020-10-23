import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


# # 圆的正投影
#
# R = input("请输入圆的半径：")
# R = float(R) # R为圆的半径
# miu = input("请输入圆的线性衰减系数：")
# miu = float(miu)
#
# x = np.linspace(-R, R, 1000) # x为投影射线向量
# p = np.linspace(0, math.pi, 1800) # p为投影射线的旋转角度
#
# a = np.arange(1800*1000, dtype=np.float64).reshape((1800, 1000))
# # 开出一个1800行（旋转1800次），1000列（投影向量共1000条）的数组，同时使用了64位浮点数提高精度
# b = np.arange(1800*1000, dtype=np.int32).reshape((1800, 1000))
# # b采用int32整数，便于后续图像处理
#
# def gray(miu, R, x): # 先计算投影值p_theta，再计算灰度值
#     p_theta = 2*miu*(np.sqrt(np.power(R, 2)-np.power(x, 2)))
#     gray_value = 100*p_theta
#     return gray_value
#
# if __name__ == '__main__':
#     for i in range(1800):
#         for j in range(1000):
#             a[i][j] = gray(miu, R, x[j])
#             b[i][j] = np.nan_to_num(a[i][j])
#     plt.imshow(b, cmap='gray')
#     plt.show()

# # 椭圆的正投影
#
# # 椭圆为x^2/a^2+y^2/b^2=1
# # 直线为y=r/sin(theta)-x*cot(theta)
#
# a = input("请输入椭圆的长轴长：")
# a = float(a)
# b = input("请输入椭圆的短轴长：")
# b = float(b)
# miu = input("请输入椭圆的线性衰减系数：")
# miu = float(miu)
#
# theta = np.linspace(0.0314, math.pi, 1800) # theta为投影射线角度,为防止除数为0时sin(theta)为0异常报错,令theta从0.0314开始
# r = np.linspace(-a, a, 5000) # r为投影射线
# m = np.zeros((1800, 5000)) # m用于存放数据,横轴为theta,纵轴为r
#
# def solve_function_coefficient(a,b,r,theta): # 联立直线与椭圆方程,求解系数
#     alpha = 1 + ((pow(a, 2))*(pow(math.cos(theta), 2)))/((pow(b, 2))*(pow(math.sin(theta), 2)))
#     beta = (-2)*(pow(a, 2)*r*math.cos(theta))/(pow(b, 2)*pow(math.sin(theta), 2))
#     gama = (pow(a, 2)*pow(r, 2))/(pow(b, 2)*pow(math.sin(theta), 2)) - pow(a, 2)
#     delta = pow(beta, 2) - 4*alpha*gama
#     # print(alpha, beta, gama, delta)
#     return alpha, beta, gama, delta
#
# def coordinate_1(alpha, beta, delta, r, theta): # 求解第一个点的坐标
#     if (delta < 0): # 当delta小于零时数学域错误报错,故令所有小于0的delta全按0处理
#         new_delta = 0
#     else:
#         new_delta = delta
#     x_1 = ((-1)*beta + math.sqrt(new_delta))/(2*alpha)
#     y_1 = r/math.sin(theta) - x_1*math.cos(theta)/math.sin(theta)
#     return x_1, y_1
#
# def coordinate_2(alpha, beta, delta, r, theta): # 求解第二个点的坐标
#     if (delta < 0):
#         new_delta = 0
#     else:
#         new_delta = delta
#     x_2 = ((-1)*beta - math.sqrt(new_delta))/(2*alpha)
#     y_2 = r/math.sin(theta) - x_2*math.cos(theta)/math.sin(theta)
#     return x_2, y_2
#
# def distance(a, b, r, theta): # 计算两点间距离
#     values = solve_function_coefficient(a, b, r, theta) # 按顺序分别为二次项系数,一次项系数,常数项,delta值
#     m_1 = coordinate_1(values[0], values[1], values[3], r, theta)
#     m_2 = coordinate_2(values[0], values[1], values[3], r, theta)
#     dis = math.sqrt(pow(m_1[0]-m_2[0], 2) + pow(m_1[1]-m_2[1], 2))
#     print(dis)
#     # if (dis > 10): # 过大值置为0
#     #     dis = 0
#     return dis
#
# if __name__ == '__main__':
#     for i in range(len(theta)):
#         for j in range(len(r)):
#             m[i][j] = 1000*distance(a, b, r[j], theta[i])
#     plt.imshow(m, cmap='gray')
#     plt.show()

