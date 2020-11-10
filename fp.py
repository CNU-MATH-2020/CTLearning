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
        if self.r is None:
            self.r = np.linspace(-1,1,256)#r 从-1到1的256个值
        if self.phi is None:
            self.phi = np.linspace(0,np.pi,180)#phi 从0到pi的180个方向
    def cbp_filter(self, projection, h):
        """
        平行束的CBP滤波
            projection : 投影数据
            h : 滤波器
        projection_cbp : 滤波后投影
        """
        [m, n] = len(self.phi), len(self.r)
        if h=="RL":                #RL滤波
            h = h_RL(n)
        elif h=="CL":              #CL滤波
            h = h_CL(n)         
        else:                      #不做滤波
            return projection 
        projection_cbp = np.zeros((m,n))  
        #卷积
        for i in range(n):
            for k in range(n):
                if 0 <= k-i+n/2 < n:
                    projection_cbp[:,i] += projection[:,k]*h[k-i+int(n/2)]
        return projection_cbp
    def rec_cbp(self, projection, N=256):
        """
        平行束的CBP重建
            projection : 投影数据
            N : 待重建图像大小. 默认 256.
        I : 重建后图像
        """
        [m, n] = len(self.phi), len(self.r)
        I = np.zeros((N,N))
        for i in range(m):         #第i个角度的投影
            c = np.cos(self.phi[i])
            s = np.sin(self.phi[i])
            for k1 in range(N):
                for k2 in range(N):
                    r=n/N*((k2-N/2)*c + (k1-N/2)*s + N/2)
                    nn = int(r)
                    t = r - nn
                    if 0<=nn<n-1:  #限定nn范围(0,n-2)
                        p = (1-t)*projection[i, nn] + t*projection[i, nn+1]#线性插值
                        I[N-1-k1, k2] += p
        return I[:,::-1]
        
class Shape:
    def draw(self, n = 256, max_x = 1, max_y = 1):
        """
        绘制原始图像
            n : 原始图像像素范围 n * n. 默认 256.
            max_x : x范围(-max_x, max_x). 默认 1.
            max_y : y范围(-max_y, max_y). 默认 1.
        I : np数组, 原始图像
        """
        I = np.zeros((n,n))            
        x = np.linspace(-max_x,max_x,n)
        y = np.linspace(-max_y,max_y,n)
        for i in range(n):
            for j in range(n):
                I[n-1-j,i] = self.point(x[i], y[j])
        return I
    def forward_projection(self, geometry: Geometry, projection: np=None):
        """
        geometry: 为扫描几何。
            对平行束扫描: 
                geometry.geom = "parallel-beam"
                geometry.r: 原点到射线的有向距离向量 r\in(-R, R)
                geometry.phi: 角度向量 phi\in[0,pi]
        projeciton: np数组，保存投影数据
        """
        m, n = len(geometry.phi), len(geometry.r)
        projection = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                projection[m-1-i, j] = self.string(geometry.r[j], geometry.phi[i])
        return projection

class Triangle(Shape):
    def __init__(self, x=[-1,1,0], y=[0,0,1], p=1):
        """
        初始化三角形参数:
            x : 三角形三个顶点的横坐标. 默认[-1,1,0]
            y : 三角形三个顶点的纵坐标. 默认[0,0,1]
            p : 线性衰减系数. 默认 1
        """
        self.x, self.y, self.p = x, y, p
       #三条直线系数ax+by+c=0
        self.a = [self.y[1]-self.y[0], self.y[2]-self.y[1], self.y[2]-self.y[0]]
        self.b = [self.x[0]-self.x[1], self.x[1]-self.x[2], self.x[0]-self.x[2]]
        self.c = []
        for i in range(3):
            self.c +=[-self.x[i]*self.a[i]-self.y[i]*self.b[i]]
        #直线与另一个点的位置关系
        self.relationship=[self.point_line(0,self.x[2],self.y[2]), 
                self.point_line(1,self.x[0],self.y[0]), self.point_line(2,self.x[1],self.y[1]) ]      
    def point_line(self, i, x, y):
        """
        判断第i条直线与点(x,y)的位置关系
        i取 0,1,2
        """
        A, B, C = self.a[i], self.b[i], self.c[i]
        d = A*x+B*y+C
        if d < 0:
            return -1
        elif d > 0:
            return 1
        else:
            return 0
    def point(self, x, y):
        """
        判断点(x,y)是否在三角形内,若p=0,则在三角形外
        """
        k = []
        for i in range(3):
            k +=[self.point_line(i, x, y)]
        if k==self.relationship:
            return  self.p
        else:
            return 0
    def string(self, r, phi):
        """
        直线(r,phi)与三角形相交弦长
        """
        c, s = np.cos(phi), np.sin(phi)
        point = set()
        for i in range(3):
            delta = self.a[i]*s - c*self.b[i]
            if delta != 0 :#有交点
                y = (self.a[i]*r + c*self.c[i])/delta
                x = -(self.c[i]*s + self.b[i]*r)/delta
                if i==0:
                    if (min(self.x[0], self.x[1]) <= x <= max(self.x[0], self.x[1]) and
                        min(self.y[0], self.y[1]) <= y <= max(self.y[0], self.y[1])):
                        point.add((x,y))
                elif i==1:
                    if (min(self.x[2], self.x[1]) <= x <= max(self.x[2], self.x[1]) and
                        min(self.y[2], self.y[1]) <= y <= max(self.y[2], self.y[1])):
                        point.add((x,y))
                else:
                    if (min(self.x[0], self.x[2]) <= x <= max(self.x[0], self.x[2]) and
                        min(self.y[0], self.y[2]) <= y <= max(self.y[0], self.y[2])):
                        point.add((x,y))
        point = list(point)
        if len(point)==2:
            p1, p2 = point[0], point[1]
            s=np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return self.p*s
        else:
            return 0

class Rectangle(Shape):
    def __init__(self, x0=0, y0=0, a=1, b=1, theta=0, p=1, transmatrix=None):
        """
        初始化矩形参数:
            (x0, y0) : 中心坐标. 默认 (0,0)
            (a, b) : 矩形的长和宽. 默认 (1,1)
            theta : 逆时针旋转角度. 默认 0
            p : 线性衰减系数. 默认 1
            transmatrix: 旋转平移矩阵
        """
        self.x0, self.y0, self.a, self.b, self.theta = x0, y0, a, b, theta
        self.p = p
        self.transmatrix = trans_matrix(self.x0, self.y0, self.theta, D=1)
    def point(self, x, y):
        """
        判断点(x,y)是否在矩形内,若p=0,则在矩形外
        """
        [x_, y_] = trans(self.transmatrix, x, y)
        p = 0
        if abs(x_)<self.a/2 and abs(y_)<self.b/2:
            p = self.p
        return p
    def string(self, r, phi):
        """
        直线(r,phi)与矩形相交弦长
        """
        dx, dy =self.a, self.b
        x0, y0 = self.x0, self.y0
        r = r - np.cos(phi)*x0 - np.sin(phi)*y0
        phi = phi - self.theta 
        s, c = np.sin(phi), np.cos(phi)
        point = []                         #记录直线与矩形交点
        if phi!=0:
            y1 = (r+c*dx/2)/s              #直线x=-dx/2
            y2 = (r-c*dx/2)/s              #直线x=dx/2
            if -dy/2 <= y1 < dy/2:
                point += [(-dx/2, y1)]
            if -dy/2 < y2 <= dy/2:
                point += [(dx/2, y2)]
        if phi!=np.pi/2:
            x1 = (r+s*dy/2)/c              #直线y=-dy/2
            x2 = (r-s*dy/2)/c              #直线y=dy/2
            if -dx/2 < x1 <= dx/2:
                point += [(x1, -dy/2)]
            if -dx/2 <= x2 < dx/2:
                point += [(x2, dy/2)]  
        if len(point)==2:
            p1, p2 = point[0], point[1]
            s=np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return self.p*s
        else:
            return 0
                
class Oval(Shape):
    def __init__(self, x0=0, y0=0, a=1, b=1, theta=0, p=1, transmatrix=None):
        """
        初始化椭圆参数:
            (x0, y0) : 中心坐标. 默认 (0,0)
            (a, b) : 椭圆的短轴与长轴. 默认 (1,1)
            theta : 逆时针旋转角度. 默认 0
            p : 线性衰减系数. 默认 1
            transmatrix: 旋转平移矩阵
        """
        self.x0, self.y0, self.a, self.b, self.theta = x0, y0, a, b, theta
        self.p = p
        self.transmatrix = trans_matrix(self.x0, self.y0, self.theta, D=1)
    def point(self, x, y):
        """
        判断点(x,y)是否在椭圆内,若p=0,则在椭圆外
        """
        [x_, y_] = trans(self.transmatrix, x, y)
        p = 0
        if (x_**2/self.a**2+y_**2/self.b**2)<1:
            p = self.p
        return p
    def string(self, r, phi):
        """
        计算直线(r,phi)与椭圆相交弦长
        """
        c, s = np.cos(phi), np.sin(phi)
        c1, s1 = np.cos(phi-self.theta), np.sin(phi-self.theta)
        a, b, x0, y0 = self.a, self.b, self.x0, self.y0
        a2, b2 = a**2, b**2
        delta=4*b2*a2*(a2*c1**2+b2*s1**2-(r-x0*c-y0*s)**2)
        if delta<0:
            return 0
        else:
            return self.p*np.sqrt(delta)/(a2*c1**2+b2*s1**2)


def trans_matrix(x0 = 0, y0 = 0, theta = 0, D = 0):
    """
    计算变换矩阵
        x0 : 水平位移. 默认 0.
        y0 : 垂直位移. 默认 0.
        theta : 逆时针旋转角度. 默认 0.
        若D不为0,则计算逆变换矩阵
    T : 三维旋转平移矩阵    
    """
    T = np.eye(3)
    c, s = np.cos(theta), np.sin(theta)
    if D == 0:
        T[0,0], T[0,1], T[0,2] = c, -s, x0
        T[1,0], T[1,1], T[1,2] = s, c, y0
    else:
        T[0,0], T[0,1], T[0,2] = c, s, -x0*c-y0*s
        T[1,0], T[1,1], T[1,2] = -s, c, x0*s-y0*c
    return T

def trans(T, x, y):
    """
    根据变换矩阵T,计算点(x,y)变换后的坐标
    """
    Z = np.ones((3,1))
    Z[0,0], Z[1,0] = x, y
    Z = np.dot(T,Z)
    return [Z[0,0],Z[1,0]]

def h_RL(n=256, d=1):
    """
    R-L滤波
    n : 一个角度下射线的数量.默认256
        d : 探测器间隔.默认1
    h : 滤波器
    """
    h = np.zeros((n,1))
    n0 = int(n/2)
    for i in range(n):
        if i%2==1:
            h[i] = -1/(np.pi*(i-n0)*d)**2
    h[n0]=1/(4*d**2)
    return h

def h_CL(n=256, d=1):
    """
    CL滤波
        n : 一个角度下射线的数量. 默认256.
        d : 探测器间隔. 默认1.
    h : 滤波器

    """
    h = np.zeros((n,1))
    n0=n/2
    for i in range(n):
        h[i] = -2/(np.pi**2*d**2*(4*(i-n0)**2-1))
    return h
        
if __name__=="__main__":    
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
    a11 = Rectangle(-0.3, 0, 0.3, 0.3, 0, 0.8)
    a12 = Rectangle(-0.5, -0.5, 0.1, 0.2, (25/180)*np.pi, 1)
    a13 = Rectangle(0.2, -0.33, 0.04, 0.05, (-25/180)*np.pi, .5)
    a14 = Rectangle(0.2, 0.4, 0.4, 0.05, (100/180)*np.pi, .8)
    a15 = Triangle([0.8, 0.83, 0.9], [0.8, 0.7, 0.91], 1)
    a16 = Triangle([-0.8, -0.83, -.09], [-0.8, -0.7, -0.91], 1)
    a17 = Triangle([0.08, 0.083, -0.09], [-0.08, 0.07, 0.091], 0.3)
    a = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17]
    
    B=ParallelBeamGeometry(np.linspace(-1,1,256), np.linspace(0,np.pi,180))                          # 平行束

    projection=a1.forward_projection(B)               # 投影
    for ai in a[1:]:
        projection += ai.forward_projection(B)
    plt.imshow(projection, cmap=plt.cm.gray)
    projection_cbp_RL=B.cbp_filter(projection,"RL")   # CBP_RL滤波
    plt.imshow(projection_cbp_RL, cmap=plt.cm.gray) 
    I_cbp_RL=B.rec_cbp(projection_cbp_RL,)             # 重建
    plt.imshow(I_cbp_RL, cmap=plt.cm.gray)     
    I=a1.draw()                                       # 原图
    for ai in a[1:]:
        I += ai.draw()
    plt.imshow(I, cmap=plt.cm.gray)


'''
def draw(*a, n = 256, max_x = 1, max_y = 1):
    #绘制椭圆序列a,返回矩阵I
    I = np.zeros((n,n))            #原始图像默认像素大小256*256
    x = np.linspace(-max_x,max_x,n)#x范围默认（-1，1）
    y = np.linspace(-max_y,max_y,n)#y范围默认（-1，1）
    for i in range(n):
        for j in range(n):
            p=0
            for ai in a:
                p += ai.point(x[i], y[j])
            I[n-1-j,i] = p
    return I
    
def p_oval(a,m=180,n=256):
    #返回椭圆的投影
    Img = np.zeros((m,n))#sino图
    phi = np.linspace(0,np.pi,m)#phi 从0到pi的m个方向
    r = np.linspace(-1,1,n)#r 从-1到1的n个值
    for i in range(m):
        for j in range(n):
            for ai in a:
                Img[m-1-i][j] += ai.string(r=r[j],phi=phi[i])
    return Img
def P_O(*a, r, phi):
    #计算图形a在直线上的投影,并返回P
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
def chongjian(Img):
    [m, n] = Img.shape
    I = np.zeros((max(m,n),max(m,n)))
    for j in range(m):  
        A = np.zeros_like(I)
        for i in range(n):
            A[i,:] = Img[j,:]
        A = rotae(A,np.pi/m*j)
        I += A
    return I

def rotae(A, theta) :   
    [m, n] = A.shape
    B = np.zeros((m,n))
    x0, y0 = np.floor(m/2), np.floor(n/2)
    T = trans_matrix(x0,y0,-theta)
    for i in range(m):
        for j in range(n):
            x, y = trans(T, i-x0, j-y0)
            xx, yy = int(round(x)), int(round(y))#线性插值
            t1 , t2 =xx-x, yy-y
            if 0<xx<m and 0<yy<n:
                B[i,j] = (1-t1)*(1-t2)*A[xx,yy]+t1*(1-t2)*A[xx-1,yy]+(1-t1)*t2*A[xx,yy-1]+t1*t2*A[xx-1,yy-1]
    return B

def cbp(Img, h):
    """
    平行束的CBP滤波
    """
    [m, n] = Img.shape
    Img_CBP=np.zeros_like(Img)
    d=1
    for i in range(n):
        for k in range(n):
            Img_CBP[:,i] += Img[:,k]*h(i-k,d)
    Img *= d
    return Img_CBP


def rec(Img,N=256):
    [m, n] = Img.shape
    I = np.zeros((N,N))
    for i in range(m):         #第i个角度的投影
        c=np.cos(np.pi/180*i)
        s=np.sin(np.pi/180*i)
        cm=N/2*(1-c-s)
        for k1 in range(N):
            for k2 in range(N):
                xm=cm+k2*c+k1*s
                nn=int(xm)
                t=xm-nn
                nn=max(0,nn)   #限定nn范围(0,n-2)
                nn=min(n-2,nn)
                p=(1-t)*Img[i,nn]+t*Img[i,nn+1]#线性插值
                I[N-1-k1,k2] += p
    return I    

import struct
fp = open('shep.jpg', 'wb')
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









               





