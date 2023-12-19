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
# %%

df = pd.read_excel('/Users/ryan/Documents/GitHub/Shenzhen Bay lab/spike_protein_3structures_markerd_with_variants_from_cncb/6vxx_variants.xls')
df_CYS = pd.read_excel('/Users/ryan/Documents/GitHub/Shenzhen Bay lab/spike_protein_3structures_markerd_with_variants_from_cncb/6vxx_variants_CYS.xls')

sub_df = df.iloc[:972, 5:10]
sub_df_CYS = df.iloc[:972, 5:10]

# Extract the x, y, z, and m columns from the subset DataFrame
x = sub_df.iloc[:972,0]
y = sub_df.iloc[:972,1]
z = sub_df.iloc[:972,2]
vn = sub_df.iloc[:972,3]
vp = sub_df.iloc[:972,4]
# %%

# 绘制直方图
plt.hist(vn, bins=10, edgecolor='black')
plt.xlabel('vn')
plt.ylabel('Frequency')
plt.title('Histogram of vn')
plt.show()

# 绘制箱线图
plt.boxplot(vn)
plt.ylabel('vn')
plt.title('Boxplot of vn')
plt.show()

# 绘制直方图
plt.hist(vp, bins=10, edgecolor='black')
plt.xlabel('vp')
plt.ylabel('Frequency')
plt.title('Histogram of vp')
plt.show()

# 绘制箱线图
plt.boxplot(vp)
plt.ylabel('vp')
plt.title('Boxplot of vp')
plt.show()
# %%

vn = np.log10(vn)
vp = np.log10(vp)

#%% Just a visulisation

# 绘制直方图
plt.hist(vn, bins=10, edgecolor='black')
plt.xlabel('vn')
plt.ylabel('Frequency')
plt.title('Histogram of log10(vn)')
plt.show()

# 绘制箱线图
plt.boxplot(vn)
plt.ylabel('vn')
plt.title('Boxplot of log10(vn)')
plt.show()

# 绘制直方图
plt.hist(vp, bins=10, edgecolor='black')
plt.xlabel('vp')
plt.ylabel('Frequency')
plt.title('Histogram of log10(vp)')
plt.show()

# 绘制箱线图
plt.boxplot(vp)
plt.ylabel('vp')
plt.title('Boxplot of log10(vp)')
plt.show()

#%% z-score normalization

col = np.arange(972)

x = np.array(x)
y = np.array(y)
z = np.array(z)

#Scale
scaler = StandardScaler()

vn = np.array(vn)
vp = np.array(vp)
vn= scaler.fit_transform(vn.reshape(-1, 1))
vp= scaler.fit_transform(vp.reshape(-1, 1))
#%% Just a visulisation

# 绘制直方图
plt.hist(vn, bins=10, edgecolor='black')
plt.xlabel('vn')
plt.ylabel('Frequency')
plt.title('Histogram of z-score log10(vn)')
plt.show()

# 绘制箱线图
plt.boxplot(vn)
plt.ylabel('vn')
plt.title('Boxplot of z-score log10(vn)')
plt.show()

# 绘制直方图
plt.hist(vp, bins=10, edgecolor='black')
plt.xlabel('vp')
plt.ylabel('Frequency')
plt.title('Histogram of z-score log10(vp)')
plt.show()

# 绘制箱线图
plt.boxplot(vp)
plt.ylabel('vp')
plt.title('Boxplot of z-score log10(vp)')
plt.show()




#%% just a visulisation for the standarization
# Create a scatter plot
plt.scatter(range(len(vn)), vn, color='blue', marker='o', label='vn')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('vn')
plt.title('Scatter Plot of standardized log10(vn)')
plt.legend()

# Show the plot
plt.show()
#%% capping

# Calculate the upper 90% value
upper = np.percentile(vn, capping_percent)

# Set the top 10% values to be equal to the upper 90% value with a random fluctuation(in order to prevent later cancelation)
for i in range(len(vn)):
    if vn[i][0] > upper:
        vn[i] = upper + 0.0001*np.random.random()

#%% just a visulisation for the capping
# Create a scatter plot
plt.scatter(range(len(vn)), vn, color='blue', marker='o', label='vn')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('vn')
plt.title('Scatter Plot of capped standardized log10(vn)')
plt.legend()

# Show the plot
plt.show()

#%%
# Create the input feature matrix
X = np.column_stack((x, y, z))
X = scaler.fit_transform(X)

#Transform X_scaled back to the original scale
X_original = scaler.inverse_transform(X)
# %%
