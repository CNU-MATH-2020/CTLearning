import math
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt #plt用于显示图片
a = input("请输入矩形的宽：")
a = float(a)
b = input("请输入矩形的长：")
b = float(b)
r = np.linspace(-a,b,100)
theta = np.linspace(0.0314, math.pi, 180)
m1 = np.zeros(180*100, dtype=np.float64).reshape((180,100))
m2 = np.zeros(180*100, dtype=np.int32).reshape((180,100))
#直线为y=(r-x*cos(theta))/sin(theta)
def grayresult(r,theta):
  c = math.cos(theta)
  s = math.sin(theta)
  x1 = r/c
  x2 = (r-b*s)/c
  y1 = r/s
  y2 = (r-a*c)/s
  l  = 0 
  if (x1 >= 0 and x1 <= a): 
      if (y1 >= 0 and y1 <= b):
          l = np.sqrt(x1**2+y1**2)
      elif(y2 >= 0 and y2 <= b):
          l = np.sqrt((a-x1)**2+y2**2)
      else:
               l = np.sqrt((x2-x1)**2+b**2)
  elif(x2 >= 0 and x2 <= a):
         if (y1 >= 0 and y1 <= b):
            l = np.sqrt(x2**2+(b-y1)**2)
         elif(y2 >= 0 and y2 <= b):
            l = np.sqrt((a-x2)**2+(b-y2)**2)
  if((y1 >= 0 and y1 <= b) and (y2 >= 0 and y2 <= b) ):
       l = np.sqrt((y2-y1)**2+a**2)
  return 100*l
if __name__ == '__main__':
    for i in range(len(theta)):
        for j in range(len(r)):
             m1[i][j]=grayresult(r[j], theta[i])    
             m2[i][j]=np.nan_to_num(m1[i][j])

plt.imshow(m2, cmap='gray')
plt.show()
  
            
             
  
  
  
  
  
  