#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt


# In[2]:


data_folder='/Users/xingyulin/Downloads/data'
data_id_number=21098501
df=pd.read_csv(f'{data_folder}/{data_id_number}.csv') #stored in panda dataframe


# In[3]:


df


# In[4]:


df['MidPrice'] = (df.Ask_Price_Level_1 + df.Bid_Price_Level_1)/2
#print(df)
log_rets = (np.log(df['MidPrice'] ) - np.log(df['MidPrice'] .shift(1))).dropna()


# # mean & variance

# In[5]:


ret_mean=np.mean(log_rets)
ret_var=np.var(log_rets)


# In[6]:


ret_mean


# In[7]:


ret_var


# # stationary test

# In[8]:


#test stationary-> And is stationary
from statsmodels.tsa.stattools import adfuller
print(adfuller(df['MidPrice']))
#since the p-value is 0.95, too big, cant reject null -> it is non-stationary
print(adfuller(log_rets))
#since the p-value is 0, can reject null -> it is stationary


# # Fit Normal

# In[9]:


#bins=42 because q**1/2
plt.hist(log_rets, bins=42)
plt.show()


# In[10]:


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate some random data to plot a histogram
#data = np.random.normal(loc=10, scale=2, size=1000)

# Fit the histogram with a Gaussian distribution
mu, std = norm.fit(log_rets)

# Plot the histogram with the fitted Gaussian distribution
plt.hist(log_rets, bins=42, density=True, stacked=True,alpha=0.6,label='Histogram')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2,label='Fitted normal distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data with Fitted Gaussian Distribution')
plt.legend()
plt.show()


# In[11]:


integral = np.trapz(p, x)
print(integral)


# # QQ plot

# In[12]:


import math
import numpy as np
from scipy.stats import lognorm
import statsmodels.api as sm
import matplotlib.pyplot as plt

#stats.probplot(log_rets, dist="norm", plot=plt)
sm.qqplot(log_rets)
plt.title('Q-Q plot for log returns')
plt.show()


# # remove outlier，fit again the normal dist

# In[13]:


log_sort=log_rets.sort_values(ascending=True)


# In[14]:


from scipy.stats import anderson
anderson(log_sort[1:1798])
#If the output statistic < critical_values, it means that the original hypothesis is accepted at the corresponding significance_level -> normal


# In[15]:


# cannot be rejected, so it actually ~ normal distribution
stats.normaltest(log_sort[1:1798])


# In[16]:


sm.qqplot(log_sort[1:1798])
plt.show()


# ## the tail may have a different distribution because the tail looks a little heavy, want to try to use power law can fit the tail, even with outlier

# In[17]:


#ccdf test, found that the tail is obviously different
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
x = np.sort(log_rets)
y = np.arange(1, len(log_rets)+1)/(len(log_rets)+1)
y = 1 - y # Calculating CCDF as rank-frequency plot
m = np.mean(x)
s = np.std(x)
c = 0.5*(1 - erf((x-m)/(s*np.sqrt(2))))
plt.figure(figsize=(15,5))
plt.subplot(1,2,2)
plt.semilogy(x,y,'b',linewidth=2,label='Log return') # CCDF
plt.semilogy(x,c,'r',linewidth=2,label='Normal')
plt.ylim([1e-4, 1])
plt.xlabel('$r$', fontsize=20)
plt.ylabel('$C(r)$', fontsize=20)
plt.tick_params(labelsize=16)
plt.title('CCDF')
plt.legend()

plt.show()


# # tail dist

# In[18]:


import numpy as np
import matplotlib.pyplot as plt

p = 0.1  # Defining tails as top p% of returns (both positive and negative)

plt.figure(2)

### Right tail ##############################################################

r = np.sort(log_rets)  # Sorting returns
r_right = r[round((1-p)*len(r)):]  # Selecting top p% of returns

N = len(r_right)  # Number of returns selected as right tail
alpha_right = N/np.sum(np.log(r_right/min(r_right)))  # Maximum-likelihood estimate for right tail exponent
# mean not defined -> sample is random

print('Right tail exponent: %4.3f' % alpha_right)

x_right = np.linspace(min(r_right), max(r_right), 100)  # Grid of points between min and max values in right tail
y_right = alpha_right*(x_right/min(r_right))**(-alpha_right-1)/min(r_right)  # Values of power law distribution on grid of points

b_right, a_right = np.histogram(r_right, 20, density=True)
centers_right = (a_right[:-1] + a_right[1:]) / 2
plt.subplot(1, 2, 1)

plt.loglog(centers_right, b_right, 'ob', markersize=8, markerfacecolor='b',label='Log return')
plt.loglog(x_right, y_right, 'r', linewidth=2,label='Normal')
plt.xlabel('$r$', fontsize=20)
plt.ylabel('$p(r)$', fontsize=20)
plt.tick_params(labelsize=16)
plt.title('Right tail')
plt.legend()


# In[19]:


r_left = r[:round(p*len(r))]  # Selecting bottom p% of returns
r_left = abs(r_left)  # Converting negative returns to positive numbers
#把左边的换成正数
#right tail exp=2.207, left tail exp=1.635, left lower, has more extreme
#behav on left
N = len(r_left)  # Number of returns selected as left tail
alpha_left = N/np.sum(np.log(r_left/min(r_left)))  # Maximum-likelihood estimate for left tail exponent

print('Left tail exponent: %4.3f' % alpha_left)

x_left = np.linspace(min(r_left), max(r_left), 100)
y_left = alpha_left*(x_left/min(r_left))**(-alpha_left-1)/min(r_left)  # Power law distribution

b_left,a_left = np.histogram(r_left,20, density=True)
#centers_left = (a_left[:-1] + a_left[1:]) / 2
plt.subplot(1, 2, 2)
plt.loglog(a_left[:-1],b_left,'ob',markersize=8, markerfacecolor='b',label='Log return')
plt.loglog(x_left,y_left,'r',linewidth=2,label='Normal')
plt.xlabel('$r$', fontsize=20)
plt.ylabel('$p(r)$', fontsize=20)
plt.tick_params(labelsize=16)
plt.title('Left tail')
plt.legend()
plt.show()


# In[20]:


df['ask_vol_sum'] = (df.Ask_Volume_Level_1 + df.Ask_Volume_Level_2 +df.Ask_Volume_Level_3+df.Ask_Volume_Level_4+
                    df.Ask_Volume_Level_5+df.Ask_Volume_Level_6+df.Ask_Volume_Level_7+df.Ask_Volume_Level_8+df.Ask_Volume_Level_9+
                    df.Ask_Volume_Level_10)
df['bid_vol_sum'] =(df.Bid_Volume_Level_1+df.Bid_Volume_Level_2+df.Bid_Volume_Level_3+df.Bid_Volume_Level_4+df.Bid_Volume_Level_5+
                   df.Bid_Volume_Level_6+df.Bid_Volume_Level_7+df.Bid_Volume_Level_8+df.Bid_Volume_Level_9+df.Bid_Volume_Level_10)
df['change_of_vol']=df['ask_vol_sum']-df['bid_vol_sum']
df['change_of_midprice']=((df['MidPrice'] ) - (df['MidPrice'] .shift(1))).dropna()
df['%change_of_vol']=(df['change_of_vol']/df['change_of_vol'].shift(1))-1
df['change_of_vol_1']=df['change_of_vol']-df['change_of_vol'].shift(1)
df['log_rets'] = (np.log(df['MidPrice'] ) - np.log(df['MidPrice'] .shift(1))).dropna()
        


# ## CORR

# In[21]:


# non-linear relationship

plt.rcParams['figure.figsize']=[12,8]
plt.rcParams.update({'font.size':16})
plt.scatter(df['log_rets'],df['change_of_vol_1'],label='change of volume vs log return')
plt.legend(loc='lower left')
plt.title('Relationship btw consecutive price levels in the LOB')
plt.ylabel('change of volume')
plt.xlabel('log return')
plt.xticks(rotation=75)
plt.show()


# In[22]:


# pearson


# In[26]:


np.corrcoef(df['change_of_vol_1'][1:], df['log_rets'][1:])


# In[27]:


#Because p is too big to reject null, we assume for now that they have no linear relationship
from scipy.stats.stats import pearsonr
pearsonr(df['log_rets'][1:], df['change_of_vol_1'][1:])


# In[24]:


# kendall&spearman


# In[25]:


#because p is small and can reject null, indicating that they are both related
print(stats.kendalltau(df['log_rets'][1:], df['change_of_vol_1'][1:]))
print(stats.spearmanr(df['log_rets'][1:], df['change_of_vol_1'][1:]))


# ## shuffle test for corr

# In[28]:


#pearson
n_permutations = 1799
permuted_r = []
for i in range(n_permutations):
    permuted_data2 = np.random.permutation(df['change_of_vol_1'][1:])
    r_perm, p_perm = pearsonr(df['log_rets'][1:], permuted_data2)
    permuted_r.append(r_perm)

# p-value
p_value = np.mean(np.abs(permuted_r) >= np.abs(r))
print(f"Permutation test p-value: {p_value:.4f}")
## no linear relationship


# In[49]:


## every time permutation will give different p-value but all smaller than 5%
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import spearmanr
corr, p = kendalltau(df['log_rets'][1:], df['change_of_vol_1'][1:])
# Randomly rearrange data2
n_permutations = 1799
permuted_corr = []
for i in range(n_permutations):
    permuted_data2 = np.random.permutation(df['change_of_vol_1'][1:])
    corr_perm, p_perm = kendalltau(df['log_rets'][1:], permuted_data2)
    permuted_corr.append(corr_perm)

# p-value
p_value = np.mean(np.abs(permuted_corr) >= np.abs(corr))
print(f"Permutation test p-value: {p_value:.4f}")


# In[64]:


import numpy as np
from scipy.stats import kendalltau
corr, p = spearmanr(df['log_rets'][1:], df['change_of_vol_1'][1:])
# Randomly rearrange data2
n_permutations = 1799
permuted_corr = []
for i in range(n_permutations):
    permuted_data2 = np.random.permutation(df['change_of_vol_1'][1:])
    corr_perm, p_perm = spearmanr(df['log_rets'][1:], permuted_data2)
    permuted_corr.append(corr_perm)

# p-value
p_value = np.mean(np.abs(permuted_corr) >= np.abs(corr))
print(f"Permutation test p-value: {p_value:.4f}")


# In[65]:


#stationary test


# In[66]:


from statsmodels.tsa.stattools import adfuller
print(adfuller(df['change_of_vol_1'][1:]))


# In[67]:


from statsmodels.tsa.stattools import adfuller
print(adfuller(df['change_of_vol_1'][1:]))
#since the p-value is 0.95, too big, cant reject null -> it is non-stationary
#print(adfuller(log_rets))
#since the p-value is 0, can reject null -> it is stationary


# ## scaling law

# In[68]:


log_rets_30_sec = (np.log(df['MidPrice'] ) - np.log(df['MidPrice'] .shift(3))).dropna()
log_rets_one_min = (np.log(df['MidPrice'] ) - np.log(df['MidPrice'] .shift(6))).dropna()
log_rets_2_min = (np.log(df['MidPrice'] ) - np.log(df['MidPrice'] .shift(12))).dropna()
log_rets_5_min = (np.log(df['MidPrice'] ) - np.log(df['MidPrice'] .shift(30))).dropna()


# In[69]:


def standardization(data):
   mu = np.mean(data, axis=0)
   sigma = np.std(data, axis=0)
   return (data - mu) / sigma
def normal_dist(x , mean , sd):
   prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
   return prob_density


# In[70]:


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mean_30 = np.mean(log_rets_30_sec)
sd_30 = np.std(log_rets_30_sec)
pdf_30 = normal_dist(log_rets_30_sec,mean_30,sd_30)
#log30=standardization(log_rets_30_sec)

mean_1min = np.mean(log_rets_one_min)
sd_1min = np.std(log_rets_one_min)
pdf_1min = normal_dist(log_rets_one_min,mean_1min,sd_1min)
#log1=standardization(log_rets_30_sec)

mean_2min = np.mean(log_rets_2_min)
sd_2min = np.std(log_rets_2_min)
pdf_2min = normal_dist(log_rets_2_min,mean_2min,sd_2min)
#log2=standardization(log_rets_30_sec)

mean_5min = np.mean(log_rets_5_min)
sd_5min = np.std(log_rets_5_min)
pdf_5min = normal_dist(log_rets_5_min,mean_5min,sd_5min)
#log5=standardization(log_rets_30_sec)

#plt.hist(log_rets_30_sec, bins=50, density=True, alpha=0.6, label='30s Intervals')
x_10s = np.linspace(log_rets_30_sec.min(), log_rets_30_sec.max(), 100)
plt.plot(x_10s, norm.pdf(x_10s, mean_30, sd_30),label='30 sec')
#plt.plot(x_10s, log30,label='30 sec')

#plt.hist(log_rets_one_min, bins=50, density=True, alpha=0.6, label='1min Intervals')
x_1min = np.linspace(log_rets_one_min.min(), log_rets_one_min.max(), 100)
plt.plot(x_1min, norm.pdf(x_1min, mean_1min, sd_1min),label='1 min')

#plt.hist(log_rets_2_min, bins=50, density=True, alpha=0.6, label='2min Intervals')
x_2min = np.linspace(log_rets_2_min.min(), log_rets_2_min.max(), 100)
plt.plot(x_2min, norm.pdf(x_2min, mean_2min, sd_2min),label='2 min')

#plt.hist(log_rets_5_min, bins=50, density=True, alpha=0.6, label='5min Intervals')
x_5min = np.linspace(log_rets_5_min.min(), log_rets_5_min.max(), 100)
plt.plot(x_5min, norm.pdf(x_5min, mean_5min, sd_5min),label='5 min')

# 添加图例和标题
plt.xlabel('log_return r', fontsize=20)
plt.ylabel('scaled frequency', fontsize=20)
plt.legend()
plt.title('Scaling law with different time horizons')
plt.show()


# In[71]:


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 计算各个时间间隔的均值和标准差
mean_30 = np.mean(log_rets_30_sec)
sd_30 = np.std(log_rets_30_sec)

mean_1min = np.mean(log_rets_one_min)
sd_1min = np.std(log_rets_one_min)

mean_2min = np.mean(log_rets_2_min)
sd_2min = np.std(log_rets_2_min)

mean_5min = np.mean(log_rets_5_min)
sd_5min = np.std(log_rets_5_min)

# 计算每个时间间隔的概率密度函数，并将其除以最大值
pdf_30 = norm.pdf(x_10s, mean_30, sd_30)/norm.pdf(mean_30, mean_30, sd_30)
pdf_1min = norm.pdf(x_1min, mean_1min, sd_1min)/norm.pdf(mean_1min, mean_1min, sd_1min)
pdf_2min = norm.pdf(x_2min, mean_2min, sd_2min)/norm.pdf(mean_2min, mean_2min, sd_2min)
pdf_5min = norm.pdf(x_5min, mean_5min, sd_5min)/norm.pdf(mean_5min, mean_5min, sd_5min)

# 将每个分布的x轴减去其均值
x_10s = np.linspace(log_rets_30_sec.min()-mean_30, log_rets_30_sec.max()-mean_30, 100)
x_1min = np.linspace(log_rets_one_min.min()-mean_1min, log_rets_one_min.max()-mean_1min, 100)
x_2min = np.linspace(log_rets_2_min.min()-mean_2min, log_rets_2_min.max()-mean_2min, 100)
x_5min = np.linspace(log_rets_5_min.min()-mean_5min, log_rets_5_min.max()-mean_5min, 100)

# 绘制图像
plt.plot(x_10s, pdf_30,label='30 sec')
plt.plot(x_1min, pdf_1min,label='1 min')
plt.plot(x_2min, pdf_2min,label='2 min')
plt.plot(x_5min, pdf_5min,label='5 min')

# 添加图例和标题
plt.xlabel('log_return r', fontsize=20)
plt.ylabel('scaled frequency', fontsize=20)
plt.legend()
plt.title('Scaling law with different time horizons')
plt.show()


# ## conditional probability

# In[72]:


## divide into two groups


# In[73]:


df_change_vol_negative = df.drop(df[df['change_of_vol_1']>0].index)
#df_change_vol_negative


# In[74]:


import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

# two subsets: change of volume>0 and <0
pos_change = df[df['change_of_vol_1'] > 0]['log_rets']
neg_change = df[df['change_of_vol_1'] < 0]['log_rets']

# pdf
kde_pos = gaussian_kde(pos_change.dropna())
kde_neg = gaussian_kde(neg_change.dropna())

# P(return|change of volume>0)&P(return|change of volume<0)
return_range = np.linspace(-0.001, 0.001, 1000)
pdf_pos = kde_pos.pdf(return_range)
pdf_neg = kde_neg.pdf(return_range)


print("P(return|change of volume>0):")
print(pdf_pos)
print("P(return|change of volume<0):")
print(pdf_neg)


# In[75]:


x_range = np.linspace(-3, 3, 1000)
plt.plot(return_range,pdf_pos)
plt.title('Condibility density given change of volume >0')
plt.xlabel('Log return value') 
plt.ylabel('Frequency') 
plt.show()


# In[76]:


plt.plot(return_range,pdf_neg)
plt.title('Conditional probability density given change of volume <0')
plt.xlabel('Log return value') 
plt.ylabel('Frequency') 
plt.show()


# In[77]:


##check if is density- integral


# In[78]:


integral = np.trapz(pdf_pos, return_range)
print(integral)


# In[81]:


sns.kdeplot(log_rets)
plt.title('log return kde')
plt.show()


# In[80]:


plt.plot(return_range,pdf_neg,label='neg VI')
plt.plot(return_range,pdf_pos,label='pos VI')
sns.kdeplot(log_rets, label='Log return')
plt.legend()
plt.title('f(Log returns) VS f(Log returns|change of volume)')
plt.xlabel('Log return value') 
plt.ylabel('Frequency') 
plt.show()


# In[83]:


df_pos=df[df['change_of_vol_1'] > 0]
df_neg=df[df['change_of_vol_1'] < 0]


# ## multivariate

# ### all data

# In[84]:


import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


# Select the variables to be modeled
variables = ['change_of_vol_1', 'log_rets']
data = df[variables].dropna().values

# 估计多元核密度函数
kde = gaussian_kde(data.T)

# 输出联合概率密度
print("Multivariate probability density:")
print(kde(data.T))


# In[85]:


# # multivariate normal
# variables = ['log_rets', 'change_of_vol_1']
# data = df[variables].dropna().values

# # estimate multivariate parameters
# mean = np.mean(data, axis=0)
# covariance = np.cov(df['change_of_vol_1'][1:],log_rets)


# data = np.random.multivariate_normal(mean, covariance)

# # fit multivariate normal distribution to data
# mean_2 = np.mean(data, axis=0)
# cov_2 = np.cov(data.T)
# dist = stats.multivariate_normal(mean_2, cov_2)

# ks_statistic, p_value = stats.kstest(data, dist.cdf)

# # print results
# print('KS statistic:', ks_statistic)
# print('p-value:', p_value)
# #


# In[102]:


import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.stats import kstest


# Select the variables to be modeled

variables = ['log_rets', 'change_of_vol_1']
data = df[variables].dropna().values

mean = np.mean(data, axis=0)
covariance = np.cov(df['change_of_vol_1'][1:],log_rets)

kde = gaussian_kde(data.T)

#ks_statistic, p_value = kstest(data.T,kde)
#print("KS Statistic: ", ks_statistic)
#print("P-value: ", p_value)


# In[88]:



xmin, ymin = data.min(axis=0)
xmax, ymax = data.max(axis=0)
x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([x.ravel(), y.ravel()])
z = np.reshape(kde(positions).T, x.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', linewidth=0)
ax.set_xlabel(variables[0], fontsize=10)
ax.set_ylabel(variables[1], fontsize=10)
ax.set_zlabel('Probability density', fontsize=10)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='z', labelsize=10)
plt.title('Multivariate distribution')
plt.show()


# ## bootstrap

# In[89]:


import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity


# In[90]:


#positive volume change
# set bootstrap parameter
n_bootstrap = 1000  # bootstrap times
sample_size = len(pos_change)  # sample size

# use bootstrap to produce n_bootstrap samples，calculate each sample's KDE
kde_bootstrap = []
for i in range(n_bootstrap):
    sample = pos_change.sample(n=sample_size, replace=True)
    # KDE
    kde_sample = gaussian_kde(sample.dropna())
    kde_bootstrap.append(kde_sample)

# bootstrap samples' KDE
pdf_bootstrap = np.zeros((n_bootstrap, len(return_range)))
for i in range(n_bootstrap):
    pdf_bootstrap[i, :] = kde_bootstrap[i].pdf(return_range)

# mean and confidence interval
pdf_mean = np.mean(pdf_bootstrap, axis=0)
pdf_std = np.std(pdf_bootstrap, axis=0)
pdf_ci_lower = pdf_mean - 1.96 * pdf_std / np.sqrt(n_bootstrap)
pdf_ci_upper = pdf_mean + 1.96 * pdf_std / np.sqrt(n_bootstrap)

# 输出结果
print("Bootstraped P(return|change of volume>0):")
print(pdf_mean)
print("95% CI Lower Bound:")
print(pdf_ci_lower)
print("95% CI Upper Bound:")
print(pdf_ci_upper)


# In[91]:


import matplotlib.pyplot as plt

def plot_kde_and_ci(pdf_pos, lower_bound, upper_bound, return_range):
    plt.plot(return_range, pdf_pos, label="KDE")
    plt.plot(return_range, lower_bound, linestyle="--", label="Bootstrap 95% CI (lower bound)")
    plt.plot(return_range, upper_bound, linestyle="--", label="Bootstrap 95% CI (upper bound)")
    plt.legend()
    plt.title('Bootstrap for positive change volume')
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.show()

def compute_pct_inside_ci(pdf_pos, lower_bound, upper_bound):
    assert len(pdf_pos) == len(lower_bound) == len(upper_bound), "Arrays should have same length"
    n_inside = 0
    for i in range(len(pdf_pos)):
        if lower_bound[i] <= pdf_pos[i] <= upper_bound[i]:
            n_inside += 1
    return n_inside / len(pdf_pos)


# In[92]:


# 95% confidence interval 
lower_bound=pdf_ci_lower
upper_bound = pdf_ci_upper

plot_kde_and_ci(pdf_pos, lower_bound, upper_bound,return_range)

pct_inside_ci = compute_pct_inside_ci(pdf_pos, lower_bound, upper_bound)


# In[94]:


from scipy.stats import ks_2samp


# Perform Kolmogorov-Smirnov test -> Unable to reject indicates consistent distribution
stat, p_value = ks_2samp(pdf_pos, pdf_mean)

print("Kolmogorov-Smirnov test statistic: {}".format(stat))
print("p-value: {}".format(p_value))


# In[95]:


#positive volume change
n_bootstrap = 1000  
sample_size = len(neg_change)  

kde_bootstrap = []
for i in range(n_bootstrap):
    sample = neg_change.sample(n=sample_size, replace=True)
    kde_sample = gaussian_kde(sample.dropna())
    kde_bootstrap.append(kde_sample)

pdf_bootstrap = np.zeros((n_bootstrap, len(return_range)))
for i in range(n_bootstrap):
    pdf_bootstrap[i, :] = kde_bootstrap[i].pdf(return_range)

# mean and Confidence interval
pdf_mean = np.mean(pdf_bootstrap, axis=0)
pdf_std = np.std(pdf_bootstrap, axis=0)
pdf_ci_lower = pdf_mean - 1.96 * pdf_std / np.sqrt(n_bootstrap)
pdf_ci_upper = pdf_mean + 1.96 * pdf_std / np.sqrt(n_bootstrap)

# results
#print("Bootstraped P(return|change of volume>0):")
#print(pdf_mean)
#print("95% CI Lower Bound:")
#print(pdf_ci_lower)
#print("95% CI Upper Bound:")
#print(pdf_ci_upper)


# In[96]:


import matplotlib.pyplot as plt

def plot_kde_and_ci(pdf_pos, lower_bound, upper_bound, return_range):
    plt.plot(return_range, pdf_pos, label="KDE")
    plt.plot(return_range, lower_bound, linestyle="--", label="Bootstrap 95% CI (lower bound)")
    plt.plot(return_range, upper_bound, linestyle="--", label="Bootstrap 95% CI (upper bound)")
    plt.legend()
    plt.title('Bootstrap for negative change volume')
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.show()

def compute_pct_inside_ci(pdf_pos, lower_bound, upper_bound):
    assert len(pdf_pos) == len(lower_bound) == len(upper_bound), "Arrays should have same length"
    n_inside = 0
    for i in range(len(pdf_pos)):
        if lower_bound[i] <= pdf_pos[i] <= upper_bound[i]:
            n_inside += 1
    return n_inside / len(pdf_pos)


# In[97]:


lower_bound=pdf_ci_lower
upper_bound = pdf_ci_upper

plot_kde_and_ci(pdf_neg, lower_bound, upper_bound,return_range)

pct_inside_ci = compute_pct_inside_ci(pdf_neg, lower_bound, upper_bound)


# In[99]:


from scipy.stats import ks_2samp


# 进行Kolmogorov-Smirnov测试——》无法拒绝说明分布一致
stat, p_value = ks_2samp(pdf_neg, pdf_mean)

# 输出结果
print("Kolmogorov-Smirnov test statistic: {}".format(stat))
print("p-value: {}".format(p_value))


# ## KS Univariate-dist

# In[101]:


import numpy as np
from scipy.stats import norm, kstest
#H0:he two data distributions agree or the data fits the theoretical distribution

returns=log_rets[18:1781]
mu, std = norm.fit(returns)
# data's CDF
ecdf = np.arange(len(returns)) / float(len(returns))
# normal's CDF
ncdf = norm.cdf(returns, mu, std)
# KS test: CDF distance
ks_statistic, p_value = kstest(ecdf, ncdf)

print("KS test statistic: {:.4f}".format(ks_statistic))
print("P-value: {:.4f}".format(p_value))


# In[ ]:





# In[ ]:





# In[ ]:




