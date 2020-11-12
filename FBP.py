import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.fftpack import fft, fftshift, ifft
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']


def padImage(img):
    """预处理图像，对图像进行填充，填充过的图像为边长是原图像对角线长度的正方形。返回值是填充后图像、原图像左上角在填充后图像中的坐标值。注意：默认左上角是(0,0)点"""

    N0, N1 = img.size  # N0，N1分别为原始图像的宽和长
    lenDiag = int(np.ceil(np.sqrt(N0 ** 2 + N1 ** 2)))  # 原图像对角线长度，即为新图像边长
    imgPad = Image.new('L', (lenDiag, lenDiag))  # 创建一个新的图像，图像中元素为长整型，边长为原始图像对角线长度
    c0, c1 = int(round((lenDiag - N0) / 2)), int(
        round((lenDiag - N1) / 2))  # 原始图像左上角在填充图像中的坐标值
    imgPad.paste(img, (c0, c1))
    return imgPad, c0, c1


def getProj(img, theta):
    """对预处理后的图像进行投影，输入待投影图像和投影射线角度，得到图像的sino图。"""

    numAngles = len(theta)  # 射线角度的数量
    sinogram = np.zeros((img.size[0], numAngles))  # sino图的大小为图像边长行，射线角度个数列

    # 开始绘制图形
    plt.ion()  # 将matplotlib转换为交互模式
    fig1, (ax1, ax2) = plt.subplots(1, 2)  # ax1，ax2两张图
    im1 = ax1.imshow(img, cmap='gray')  # ax1为物体图像
    ax1.set_title('投影图像')  # ax1标题
    """ax2为投影图像，aspect='auto'表示根据画布大小自动调整单元格大小，extent指定x轴和y轴的极值(前两个数是x轴极小和极大值，后两个数是y轴极大和极小值)。注意：默认图像左上角为(0,0)点"""
    im2 = ax2.imshow(sinogram, extent=[theta[0], theta[-1], img.size[0] - 1, 0], cmap='gray', aspect='auto')
    ax2.set_xlabel('旋转角度')  # ax2的x轴名
    ax2.set_title('正弦图')  # ax2的标题
    plt.show()

    # 获取投影数据
    for n in range(numAngles):  # 旋转射线
        rotImgObj = img.rotate(90 - theta[n], resample=Image.BICUBIC)  # 旋转角度为(pi/2~-pi/2)，插值方法为双三次插值
        im1.set_data(rotImgObj)
        sinogram[:, n] = np.sum(rotImgObj, axis=0)  # 在sinogram矩阵的第n列存入axis=0表示压缩行，即将每一列元素相加，矩阵被压缩为一行，相当于对物体做投影
        im2.set_data(Image.fromarray((sinogram - np.min(sinogram)) / np.ptp(sinogram) * 255))  # 将像素值映射到0~255
        fig1.canvas.draw()
        fig1.canvas.flush_events()

    plt.ioff()
    return sinogram  # 返回值为投影矩阵


def projFilter(sino):
    """对投影结果进行Fourier变换、滤波、逆Fourier变换，输入值为投影结果矩阵sino[n,phi]，n为投影数，phi为角度数。输出结果为Fourier变换、滤波、逆Fourier变换后结果。"""

    projLen, numAngles = sino.shape  # projLen为投影数，numAngles为角度数
    step = 2 * np.pi / projLen  # 步长
    w = np.arange(-np.pi, np.pi, step)
    w = abs(w)
    filt = fftshift(w)  # 移动零频点到频谱中间
    filtSino = np.zeros((projLen, numAngles))
    for i in range(numAngles):
        projfft = fft(sino[:, i])  # 对投影数据做Fourier变换
        filtProj = projfft * filt  # 进行频域滤波
        filtSino[:, i] = np.real(ifft(filtProj))  # 滤波结果做逆Fourier变换
    return filtSino


def backproject(sinogram, theta):
    """反投影函数。输入为滤波后numpy数组，行为投影数，列为投影角度数；theta为投影角度。输出为反投影后的二维numpy数组。"""

    imageLen = sinogram.shape[0]  # 重建后图像的边长即为投影数
    reconMatrix = np.zeros((imageLen, imageLen))  # 重建后图像矩阵
    x = np.arange(imageLen) - imageLen / 2  # 创建以(0,0)为中心的坐标系
    y = x.copy()
    X, Y = np.meshgrid(x, y)  # 构建网格

    # 开始绘制图形
    plt.ion()  # 更改为交互模式
    fig2, ax = plt.subplots()
    im = plt.imshow(reconMatrix, cmap='gray')

    theta = theta * np.pi / 180
    numAngles = len(theta)

    for n in range(numAngles):
        Xrot = X * np.sin(theta[n]) - Y * np.cos(theta[n])
        # 网格形式下绕原点的旋转的X坐标
        XrotCor = np.round(Xrot + imageLen / 2)
        # 移动原始图像坐标并取整。
        XrotCor = XrotCor.astype('int')
        projMatrix = np.zeros((imageLen, imageLen))
        m0, m1 = np.where((XrotCor >= 0) & (XrotCor <= (imageLen - 1)))
        # 旋转后，会有超出原始坐标大小的新坐标，因此需要进行一次判断。
        s = sinogram[:, n]  # 得到投影
        projMatrix[m0, m1] = s[XrotCor[m0, m1]]
        reconMatrix += projMatrix
        im.set_data(Image.fromarray((reconMatrix - np.min(reconMatrix)) / np.ptp(reconMatrix) * 255))  # 像素值映射到0~255
        ax.set_title('角度为 %.2f °' % (theta[n] * 180 / np.pi))
        fig2.canvas.draw()
        fig2.canvas.flush_events()

    plt.close()
    plt.ioff()
    backprojArray = np.flipud(reconMatrix)  # 对矩阵进行翻转，使得成像为正。
    return backprojArray


if __name__ == '__main__':
    myImg = Image.open('shepp-logan(256).png').convert('L')  # convert('L')表示每个像素用8bit表示，0表示黑，255表示白
    myImgPad, c0, c1 = padImage(myImg)  # 新图像
    dTheta = 1  # 投影射线角度改变
    theta = np.arange(0, 181, dTheta)  # 投影射线角度
    print('正在投影\n')
    mySino = getProj(myImgPad, theta)  # mySino为投影矩阵
    print('正在变换\n')
    filtSino = projFilter(mySino)  # 对投影结果进行Fourier变换、滤波、逆Fourier变换
    print('正在反投影')

    recon = backproject(filtSino, theta)
    recon2 = np.round((recon - np.min(recon)) / np.ptp(recon) * 255)  # 将像素值映射到0~255
    reconImg = Image.fromarray(recon2.astype('uint8'))
    n0, n1 = myImg.size
    reconImg = reconImg.crop((c0, c1, c0 + n0, c1 + n1))

    fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.imshow(myImg, cmap='gray')
    ax1.set_title('原始图像')
    ax2.imshow(reconImg, cmap='gray')
    ax2.set_title('投影图像')
    ax3.imshow(ImageChops.difference(myImg, reconImg), cmap='gray')
    ax3.set_title('误差')
    plt.show()
