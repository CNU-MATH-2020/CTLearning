import numpy as np
import matplotlib.pyplot as plt


class Geometry:
    def __init__(self):
        self.geom = None
    
class ParallelBeamGeometry(Geometry):
    def __init__(self, r=None, phi=None):
        self.geom = "parallel-beam"
        self.r = r
        self.phi = phi
G=ParallelBeamGeometry([-1,0,1],0)#创建平行光束对象r=0,phi=0

class Shape:
    def forward_projeciton(self, geometry:Geometry, projection:np=None):
        pass

class Oval(Shape):
    def __init__(self, x0, y0, a, b, theta, p, transmatrix=None):
        """
        (x0, y0): 中心坐标
        theta: 逆时针旋转角度
        (a, b): 椭圆的短轴与长轴
        p:密度
        transmatrix:旋转平移矩阵
        """
        self.x0, self.y0, self.a, self.b, self.theta = x0, y0, a, b, theta
        self.p = p
        self.transmatrix = trans_matrix(self.x0, self.y0, self.theta)
    def midu(self, x, y):
        '''计算点(x,y)是否在椭圆内,若p=0,则在椭圆外'''
        [x_,y_] = trans(self.transmatrix, x, y)
        p = 0;
        if (x_**2/self.a**2+y_**2/self.b**2)<1:
            p += self.p
        return p
    def string(self, r, phi):
        '''用直线r,phi求与椭圆相交弦长'''
        c, s = np.cos(phi), np.sin(phi)
        a, b, x0, y0 = self.a, self.b, self.x0, self.y0
        a2, b2 = a**2, b**2
        delta = (c*s*(a2-b2)*r-b2*s*x0+a2*c*y0)**2-(b2*s**2+a2*c**2)*(b2*c**2*r**2+a2*s**2*r**2-a2*b2+b2*x0**2+a2*y0**2+2*b2*c*r*x0+2*a2*s*r*y0)
        if delta<0:
            return 0
        else:
            return self.p*np.sqrt(delta)/(a2*c**2+b2*s**2)
        
    def forward_projection(self, geometry: Geometry, projection:np=None):
        """
        geometry: 为扫描几何。
            对平行束扫描:
                geometry.geom = "parallel-beam"
                geometry.r: 原点到射线的有向距离向量 r\in(-R, R)
                geometry.phi: 角度向量 phi\in[0,pi]
        projeciton: np数组，保存投影数据
        """

def trans_matrix(x0 = 0, y0 = 0, theta = 0, D = 0):
    '''计算变换矩阵tran，水平位移x0，垂直位移y0，逆时针旋转角度theta
    若D不为0,则计算逆变换矩阵'''
    tran = np.eye(3)
    c, s = np.cos(theta), np.sin(theta)
    if D == 0:
        tran[0,0], tran[0,1], tran[0,2] = c, -s, x0
        tran[1,0], tran[1,1], tran[1,2] = s, c, y0
    else:
        tran[0,0], tran[0,1], tran[0,2] = c, s, -x0*c-y0*s
        tran[1,0], tran[1,1], tran[1,2] = -s, c, x0*s-y0*c
    return tran

def trans(T, x, y):
    '''变换矩阵T,计算变换后点的坐标'''
    Z = np.ones((1,3)).T
    Z[0,0], Z[1,0] = x, y
    Z = np.dot(T,Z)
    return [Z[0,0],Z[1,0]]

def draw(*a, n = 256, max_x = 1, max_y = 1):
    '''绘制椭圆序列a,返回矩阵I'''
    I = np.zeros((n,n))            #原始图像默认像素大小256*256
    x = np.linspace(-max_x,max_x,n)#x范围默认（-1，1）
    y = np.linspace(-max_y,max_y,n)#y范围默认（-1，1）
    for i in range(n):
        for j in range(n):
            p=0
            for ai in a:
                p += ai.midu(x[i], y[j])
            I[i,j] = p
    plt.imshow(I, cmap=plt.cm.gray)
    return I

def P_O(*a, r, phi):
    '''计算图形a在直线上的投影,并返回P'''
    t1, t2, n = -1, 1, 1000 #直线上参数t的范围,可以取长轴长
    dt = (t2-t1)/n 
    P = 0
    for t in np.linspace(t1, t2, n):
        x = r*np.cos(phi)-np.sin(phi)*t
        y = r*np.sin(phi)+np.cos(phi)*t
        p = 0
        for ai in a:
            p += ai.midu(x, y)
        P += p*dt
    return P

def rotae(A, theta) :   
    [m, n] = A.shape
    B = np.zeros((m,n))
    x0, y0 = np.floor(m/2), np.floor(n/2)
    T = trans_matrix(x0,y0,-theta)
    for i in range(m):
        for j in range(n):
            x, y = trans(T, i-x0, j-y0)
            x, y = int(round(x)), int(round(y))
            if 0<=x<m and 0<=y<n:
                B[i,j] = A[x,y]
    return B

if __name__=="__main__":
    import datetime
    starttime = datetime.datetime.now()
    
    a1 = Oval(0, 0, 0.69, 0.92, 0, 1)
    a2 = Oval(0, -0.0184, 0.6624, 0.874, 0, -0.8)
    a3 = Oval(0.22, 0, 0.11, 0.31, (-18/180)*np.pi, -0.2)
    a4 = Oval(-0.22, 0, 0.16, 0.41, (18/180)*np.pi,-0.2)
    a5 = Oval(0, 0.35, 0.21, 0.25, 0, 0.1)
    a6 = Oval(0, 0.1, 0.046, 0.046, 0, 0.1)
    a7 = Oval(0, -0.1, 0.046, 0.046, 0, 0.1)
    a8 = Oval(-0.08, -0.605, 0.046, 0.023, 0, 0.1)
    a9 = Oval(0, -0.606, 0.023, 0.023, 0, 0.1)
    a10 = Oval(-0.06, -0.605, 0.023, 0.046, 0, 0.1)
    
    a = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
    I = draw(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
    Img = np.zeros((256,256))#sino图
    phi = np.linspace(0,np.pi,256)#phi 从0到pi的180个方向
    r = np.linspace(-1,1,256)#r 从-1到1的100个值
    for i in range(256):
        for j in range(256):
            for ai in a:
                Img[i][j] += ai.string(r=r[j],phi=phi[i])
    plt.imshow(Img,cmap=plt.cm.gray)
    endtime = datetime.datetime.now()
    print('投影时间:',endtime-starttime)
    
    starttime = datetime.datetime.now()   
    B = np.zeros((256,256))
    for j in range(256):  
        A = np.zeros((256,256))
        for i in range(256):
            A[i,:] = Img[j,:]
        A = rotae(A,np.pi/256*j)
        B += A
    
    plt.imshow(rotae(B,np.pi/1), cmap=plt.cm.gray)
    endtime = datetime.datetime.now()
    print("重建时间:",endtime-starttime)
'''
import struct
fp = open('Imgc.jpg', 'wb')
write_buf = struct.pack('={}f'.format(Img.size), *Img.flatten())
fp.write(write_buf)
fp.close()
#读取二进制文件
fp = open('Imgc.raw', 'rb')
raw_data = fp.read(180 * 100 * 4)
raw_data = struct.unpack('={}f'.format(180 * 100), raw_data)
image = np.asarray(raw_data).reshape(180, 100)
fp.close()
plt.imshow(image,cmap=plt.cm.gray)
S=[]
a=Oval(0,0,1,1,0,1)
for r in np.linspace(-1,1,100):
    S+=[a.string(r, np.pi/2)]
plt.plot(np.linspace(-1,1,100),S)
'''






