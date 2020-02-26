##数据导入及预处理
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import  stats
import statsmodels.api as sm  # 统计相关的库
import arch  # 条件异方差模型相关的库

IndexData = DataAPI.MktIdxdGet(indexID=u"",ticker=u"000001",beginDate=u"20140603",endDate=u"20181231",field=u"tradeDate,closeIndex,CHGPct",pandas="1") 
IndexData = IndexData.set_index(IndexData['tradeDate'])

IndexData.plot(subplots=True,figsize=(18,12))  #作出上证指数及收益率的序列图

## ADF检验
data = IndexData['CHGPct'] # 上证指数日收益率
t = sm.tsa.stattools.adfuller(data)  
print "p-value:   ",t[1]

##AR模型拟合
temp = np.array(data) # 载入收益率序列
model = sm.tsa.AR(temp)  
results_AR = model.fit()  
plt.figure(figsize=(10,4))
plt.plot(temp,'b',label='CHGPct')
plt.plot(results_AR.fittedvalues, 'r',label='AR model')
plt.legend()

print len(results_AR.roots) #判定阶数

##单位根检验
pi,sin,cos = np.pi,np.sin,np.cos
r1 = 1
theta = np.linspace(0,2*pi,360)
x1 = r1*cos(theta)
y1 = r1*sin(theta)
plt.figure(figsize=(6,6))
plt.plot(x1,y1,'k')  # 画单位圆
roots = 1/results_AR.roots  # 注意，这里results_AR.roots 是计算的特征方程的解，特征根应该取倒数

for i in range(len(roots)):
    plt.plot(roots[i].real,roots[i].imag,'.r',markersize=8)  #画特征根
plt.show()

##计算拟合优度Adj R-Squared
delta = results_AR.fittedvalues  - temp[22:]  # 残差
score = 1 - delta.var()/temp[22:].var()
print score

##作偏自相关系数图
fig = plt.figure(figsize=(20,5))
ax1=fig.add_subplot(111)
fig = sm.graphics.tsa.plot_pacf(temp,lags=50,ax=ax1)

##运用AIC准则给ARMA模型定阶
sm.tsa.arma_order_select_ic(data,max_ar=6,max_ma=6,ic='aic')['aic_min_order'] 

##计算残差，作出残差序列图 
at = data -  data.mean()
at2 = np.square(at)
plt.figure(figsize=(10,6))
plt.subplot(211)
plt.plot(at,label = 'at')
plt.legend()
plt.subplot(212)
plt.plot(at2,label='at^2')
plt.legend(loc=0)

##混成检验
m = 12 # 我们检验12个自相关系数
acf,q,p = sm.tsa.acf(at2,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
out = np.c_[range(1,13), acf[1:], q, p]
output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
output = output.set_index('lag')
output

##通过残差的偏自相关系数判断ARCH模型阶次
fig = plt.figure(figsize=(20,5))
ax1=fig.add_subplot(111)
fig = sm.graphics.tsa.plot_pacf(at2,lags = 30,ax=ax1)

##建立ARCH模型
train = data[:-143]
test = data[-143:]
am = arch.arch_model(train,mean='Constant',vol='ARCH',p=5) 
res = am.fit()

res.summary()

##建立GARCH模型
train = data[:-143]
test = data[-143:]
am = arch.arch_model(train,mean='Constant',vol='GARCH') 
res = am.fit()

res.summary()
