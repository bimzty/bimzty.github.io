#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:12:00 2023

@author: ryan
"""

#%%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as colors
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, Matern
import random
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math
from sklearn.gaussian_process.kernels import RBF, Matern,  ExpSineSquared, DotProduct
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RationalQuadratic,WhiteKernel
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import SMOTERegression
from sklearn.ensemble import RandomForestRegressor

#%%
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print ('neighbors'),neighbors
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
a=np.array([[1,2,3],[4,5,6],[2,3,1],[2,1,2],[2,3,4],[2,3,4]])
s=Smote(a,N=100)
print (s.over_sampling())

#%%

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 生成一个不平衡的样本数据集
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3,
                           n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=42)

# 绘制原始数据和过采样后的数据
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', label='Original Data')
#plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, marker='x', label='Resampled Data')
plt.legend()
plt.title('SMOTE Over-sampling')
plt.show()

# 创建 SMOTE 对象
smote = SMOTE(sampling_strategy='auto', random_state=42)

# 使用 SMOTE 进行过采样
X_resampled, y_resampled = smote.fit_resample(X, y)

# 绘制原始数据和过采样后的数据
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', label='Original Data')
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, marker='x', label='Resampled Data')
plt.legend()
plt.title('SMOTE Over-sampling')
plt.show()

#%%开始

#Parameter change
#纯手工K-Fold section K：

df = pd.read_excel('/Users/ryan/Documents/GitHub/Shenzhen Bay lab/spike_protein_3structures_markerd_with_variants_from_cncb/6vxx_variants.xls')
df_CYS = pd.read_excel('/Users/ryan/Documents/GitHub/Shenzhen Bay lab/spike_protein_3structures_markerd_with_variants_from_cncb/6vxx_variants_CYS.xls')

sub_df = df.iloc[:972, 5:10]
sub_df_CYS = df.iloc[:972, 5:10]

# Extract the x, y, z, and m columns from the subset DataFrame
x = sub_df.iloc[:972,0]
y = sub_df.iloc[:972,1]
z = sub_df.iloc[:972,2]
vp = sub_df.iloc[:972,4]

# Create the label array based on the conditions
#labels = np.where(vp > 0.5, 2, np.where((vp > 0.1) & (vp <= 0.5), 1, 0))

labels = np.where(vp > 0.5, 3, 
                  np.where((vp > 0.3) & (vp <= 0.5), 2,
                  np.where((vp > 0.1) & (vp <= 0.3), 1, 0)))


X = np.column_stack((x, y, z))

# Initialize SMOTE with the desired sampling strategy
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Perform SMOTE oversampling
X_resampled, labels_resampled = smote.fit_resample(X, labels)

#we discover that the added oversampling values are all behind the original values
matching_indices = np.where((X[:, None] == X_resampled).all(-1))[1]

vp_resampled = []
for i in X_resampled:
    
    if i in X:
        
        indices = (np.where(np.all(X == i, axis=1))[0])[0]
        
        vp_resampled.append(vp[indices])
    
    else:
        
        indices = (np.where(np.all(X_resampled == i, axis=1))[0])[0]
        
        # if labels_resampled[indices] == 2:
            
        #     vp_resampled.append(np.random.uniform(0.5, 1))
        
        # if labels_resampled[indices] == 1:
            
        #     vp_resampled.append(np.random.uniform(0.1, 0.5))
        
        # if labels_resampled[indices] == 0:
            
        #     vp_resampled.append(np.random.uniform(0, 0.1))
            
        if labels_resampled[indices] == 3:
            
            vp_resampled.append(np.random.uniform(0.5, 1))
        
        if labels_resampled[indices] == 2:
            
            vp_resampled.append(np.random.uniform(0.3, 0.5))
        
        if labels_resampled[indices] == 1:
            
            vp_resampled.append(np.random.uniform(0.1, 0.3))
        
        if labels_resampled[indices] == 0:
            
            vp_resampled.append(np.random.uniform(0, 0.1))
            
     
vp_resampled = np.array(vp_resampled)
        
#%% Cross validation find best gp

score = -1000

for size in [0.05]:
    
    # Splitting the dataset into training and test sets
    # Random state is currently fixed
    #seed = random.randint(1, 100)
    seed = 20
    
    X_train, X_test, vp_train, vp_test = train_test_split(X_resampled, vp_resampled, test_size=size, random_state= seed)

#we discover that the added oversampling values are all behind the original values
vp_selected = vp[vp > 0.2]
X_selected = X[vp_selected.index]
matching_indices = np.where((X_selected[:, None] == X_train).all(-1))[1]
len(matching_indices)
#%%
#Rational Quadratic best: size = 0.1,seed = 20, alpha = 1, length_scale = 1
for K_num in [10]:
    for l in [1]:
        for alpha_num in [1]:
            
            ker = RationalQuadratic(length_scale= l ,alpha=alpha_num,length_scale_bounds=(1e-100,1e100))  # You can choose other kernels as well
            
            gp = GaussianProcessRegressor(kernel=ker,
                                            optimizer='fmin_l_bfgs_b',  # Use L-BFGS-B optimizer
                                            n_restarts_optimizer=3) 
            
            #gp.fit(X_train, vp_train_noi)
            gp.fit(X_train,vp_train)
            
            print(f"The model is {gp}")
            print(f"The K-Fold is {K_num}")
            print(f"The size is {size}")
            
            
            #neg_mean_squared_error --> super smart way, since most optimization target at maximizing the score, 
            #but we want to minimize the mean_square error, which is the same as maximizing the negative mean squared error
            
            #temp_score = np.mean(cross_val_score(gp, X_train, vp_train, cv= K_num, scoring='neg_mean_squared_error'))
            
            temp_score = gp.log_marginal_likelihood()
            
            if score < temp_score:
                    
                best_gp = gp
                score = temp_score
                K = K_num

#%% Take into test set

vp_pred,sigma = best_gp.predict(X_test,return_std=True)
#vp_pred = best_rf.predict(X_test)


mse = mean_squared_error(vp_test, vp_pred)
#mse_noi = mean_squared_error(vp_test_noi, vp_pred)
r2 = r2_score(vp_test, vp_pred)

print(f"The mse is {mse}, this is the mse between vp_pred and vp_test")
#print(f"The mse_noi is {mse_noi},this is the mse between vp_pred and vp_test_noi")
print(f"The R-squared is {r2}")

print(f"The seed is {seed}")
print(f"The best_model is {best_gp}")
print(f"The log_marginal_likelihood is {score}")
#print(f"The best_model is {best_rf}")
print(f"The best_cross_val K-Fold is {K}")
print(f"The best_size is {size}")
#print(f"The noise is mu={mu}, std={std}")

plt.hist(vp, bins=20, edgecolor='black')
plt.xlabel('Log 10 Virus Number')
plt.ylabel('Frequency')
plt.title('Distribution of Virus Number')
plt.show()
count = np.sum(vp_test > 0.2)
print("Number of observations in vp where vp_test > 0.5:", count)







#%% Plot

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot for vp
axs[0].scatter(range(len(vp_test)), vp_test, color='blue', marker='o')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('vp')
axs[0].set_title('Scatter Plot of vp_test')

# Scatter plot for vp_pred

axs[1].scatter(range(len(vp_pred)), vp_pred, color='blue', marker='o')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('vp_pred')
axs[1].set_title('Scatter Plot of vp_pred')

# Add a main title
plt.suptitle(f"Test pred on test data", fontsize=16)

plt.tight_layout()
plt.show()



#%%

fig, ax = plt.subplots() # 创建图实例
ax.errorbar(range(len(vp_pred)), vp_pred.ravel(), yerr=sigma,ecolor='r', color='b', fmt='o', label='Uncertainty')
ax.scatter(range(len(vp_test)), vp_test, color='g', marker='o', label='vp_test')

ax.set_xlabel('Index')
ax.set_ylabel('vp_pred')
ax.set_title('Predicted vp with Uncertainty')
ax.legend()
plt.show()

#%% Take into original non-resampled set

vp_pred,sigma = gp.predict(X,return_std=True)
#vp_pred = best_rf.predict(X_test)

mse = mean_squared_error(vp, vp_pred)
#mse_noi = mean_squared_error(vp_test_noi, vp_pred)
r2 = r2_score(vp, vp_pred)

print(f"The mse is {mse}, this is the mse between vp_pred and vp_test")
#print(f"The mse_noi is {mse_noi},this is the mse between vp_pred and vp_test_noi")
print(f"The R-squared is {r2}")

print(f"The seed is {seed}")
print(f"The best_model is {gp}")
#print(f"The best_model is {best_rf}")
print(f"The best_cross_val K-Fold is {K}")
print(f"The best_size is {size}")
#print(f"The noise is mu={mu}, std={std}")

plt.hist(vp, bins=20, edgecolor='black')
plt.xlabel('Log 10 Virus Number')
plt.ylabel('Frequency')
plt.title('Distribution of Virus Number')
plt.show()
count = np.sum(vp_test > 0.2)
print("Number of observations in vp where vp_test > 0.5:", count)

#%% Plot

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot for vp
axs[0].scatter(range(len(vp)), vp, color='blue', marker='o')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('vp')
axs[0].set_title('Scatter Plot of vp_test')

# Scatter plot for vp_pred

axs[1].scatter(range(len(vp_pred)), vp_pred, color='blue', marker='o')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('vp_pred')
axs[1].set_title('Scatter Plot of vp_pred')

# Add a main title
plt.suptitle(f"Test pred on test data", fontsize=16)

plt.tight_layout()
plt.show()

#%%

fig, ax = plt.subplots() # 创建图实例
ax.errorbar(range(len(vp_pred)), vp_pred.ravel(), yerr=sigma,ecolor='r', color='b', fmt='o', label='Uncertainty')

ax.scatter(range(len(vp)), vp, color='g', marker='o', label='vp_test')

ax.set_xlabel('Index')
ax.set_ylabel('vp_pred')
ax.set_title('Predicted vp with Uncertainty')
ax.legend()
plt.show()
