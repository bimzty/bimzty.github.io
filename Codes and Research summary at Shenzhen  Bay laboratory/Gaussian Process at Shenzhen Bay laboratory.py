#%%START
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, DotProduct, RationalQuadratic, WhiteKernel
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

# File path should be changed
file_path = r"D:\Desktop\学习\2023深圳湾实习\code poll\6vxx__A.xlsx"

try:
    # Read data from Excel file
    df = pd.read_excel(file_path)
    
    # Extract columns
    index = df['resID']
    x = df['x']
    y = df['y']
    z = df['z']
    num = df['num']
    
    # Concatenate columns into a single variable
    data = pd.concat([x, y, z, num], axis=1)
    print(data)
    
    # Concatenate x, y, z columns into a single variable X
    X = pd.concat([x, y, z], axis=1)
    
    # Assign the 'num' column to variable y
    Y = num
    print(Y)

except FileNotFoundError:
    print("File not found! Please check the file path.")
except IOError:
    print("Unable to read the file!")
#%% Data exploration
#1.histogram for values
import numpy as np
import matplotlib.pyplot as plt

# x histogram
plt.hist(x, bins=50, color='blue', alpha=0.5)
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Histogram of x')
plt.show()

# y histogram
plt.hist(y, bins=50, color='green', alpha=0.5)
plt.xlabel('y')
plt.ylabel('Frequency')
plt.title('Histogram of y')
plt.show()

# z histogram
plt.hist(z, bins=50, color='red', alpha=0.5)
plt.xlabel('z')
plt.ylabel('Frequency')
plt.title('Histogram of z')
plt.show()

# histogram for VirusNumber
plt.hist(num, bins=50, color='purple', alpha=0.5)
plt.xlabel('num')
plt.ylabel('Frequency')
plt.title('Histogram of VN')
plt.show()

#2.Mutation in 3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# six axis
view_angles = [(30, 45), (0, 90), (60, 30), (45, 0), (30, 60), (0, 0)]

fig, axs = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={'projection': '3d'})
axs = axs.flatten()

# plot
for i in range(len(view_angles)):
    ax = axs[i]
    ax.scatter(x, y, z, c=num, cmap='coolwarm', vmin=np.min(num), vmax=np.max(num))
    ax.view_init(elev=view_angles[i][0], azim=view_angles[i][1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'View {i+1}')

plt.tight_layout()

plt.show()

#3.Heatmap/density plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(np.array(x).flatten(), np.array(y).flatten(), bins=10)
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
zpos = np.zeros_like(xpos)
dx = dy = 0.1 * np.ones_like(zpos)
dz = hist.flatten()
ax.bar3d(xpos.ravel(), ypos.ravel(), zpos.ravel(), dx.ravel(), dy.ravel(), dz, cmap='coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Num')
plt.title('Heatmap: Protein Residue Positions and Num')
plt.show()

#4. Virogram(Error due to the package)
# data = np.column_stack((x, y, z, num))
# variogram = Variogram(data, n_lags=10)
# variogram.plot()

# plt.xlabel('Lag Distance')
# plt.ylabel('Variogram')
# plt.title('Variogram: Relationship between X, Y, Z, and Num')
# plt.show()
#%% Data Preprocessing
# log transformation of y
y_log = np.log10(y)

plt.hist(y, bins=30, density=True, alpha=0.5)
plt.title("Histogram of y")
plt.show()

import scipy.stats as stats
# Plot the transformed y Q-Q plot
stats.probplot(y_log, dist="norm", plot=plt, fit=True)
plt.title("Q-Q Plot of Transformed y")
plt.show()

# Normality test using several methods
# Normality test - Kolmogorov-Smirnov test
kstest_result = stats.kstest(y_log, 'norm')
print("Kolmogorov-Smirnov test p-value:", kstest_result.pvalue)

# Normality test - Shapiro-Wilk test
shapiro_result = stats.shapiro(y_log)
print("Shapiro-Wilk test p-value:", shapiro_result.pvalue)

# Normality test - Anderson-Darling test
anderson_result = stats.anderson(y_log)
print("Anderson-Darling test statistic:", anderson_result.statistic)
print("Anderson-Darling test critical values:", anderson_result.critical_values)

# Normality test
p_value = stats.normaltest(y)[1]
alpha = 0.05

if p_value < alpha:
    print("The y variable does not follow a normal distribution.")
else:
    print("The y variable follows a normal distribution.")
#%%Box-Cox for further transform y
# we do not require normal distribution in the GP, so it is just a test here.

from scipy.stats import boxcox

# Reshape y to be 1-dimensional
y = np.reshape(y, -1)

# Perform Box-Cox transformation
y_transformed, lambda_ = boxcox(y + 3)

# Plot the transformed y histogram
plt.hist(y_transformed, bins=30, density=True, alpha=0.5)
plt.title("Histogram of Transformed y")
plt.show()

# Plot the transformed y Q-Q plot
stats.probplot(y_transformed, dist="norm", plot=plt, fit=True)
plt.title("Q-Q Plot of Transformed y")
plt.show()

# Same test for normality
kstest_result = stats.kstest(y_transformed, 'norm')
print("Kolmogorov-Smirnov test p-value:", kstest_result.pvalue)

shapiro_result = stats.shapiro(y_transformed)
print("Shapiro-Wilk test p-value:", shapiro_result.pvalue)

anderson_result = stats.anderson(y_transformed)
print("Anderson-Darling test statistic:", anderson_result.statistic)
print("Anderson-Darling test critical values:", anderson_result.critical_values)

# Test for normality
p_value = stats.normaltest(y_transformed)[1]
alpha = 0.05

if p_value < alpha:
    print("The transformed y variable does not follow a normal distribution.")
else:
    print("The transformed y variable follows a normal distribution.")

#uniform variable name for convinience (better split next time)
y_scaled = y_transformed
y = y_transformed
#%%Feature processing: Feature extraction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

# Feature Engineering: Feature extraction
center_x = data['x'].mean()
center_y = data['y'].mean()
center_z = data['z'].mean()

data['distance_to_center'] = np.sqrt((data['x'] - center_x)**2 + (data['y'] - center_y)**2 + (data['z'] - center_z)**2)
data['mean_x'] = center_x
data['mean_y'] = center_y
data['mean_z'] = center_z
data['diff_x'] = data['x'] - center_x
data['diff_y'] = data['y'] - center_y
data['diff_z'] = data['z'] - center_z
data['sum_xyz'] = data['x'] + data['y'] + data['z']
data['std_x'] = data['x'].std()
data['std_y'] = data['y'].std()
data['std_z'] = data['z'].std()
data['volume'] = (data['x'].max() - data['x'].min()) * (data['y'].max() - data['y'].min()) * (data['z'].max() - data['z'].min())

# Feature Engineering from polynomial
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = ['x', 'y', 'z']  
X_poly = poly.fit_transform(data[poly_features])
poly_columns = poly.get_feature_names_out(poly_features)
X_poly_df = pd.DataFrame(X_poly, columns=poly_columns)  # Create a DataFrame for the polynomial features

# Print the polynomial feature column names
print("Polynomial Feature Columns:", poly_columns)

# Construct variables
X = pd.concat([data[['x', 'y', 'z', 'distance_to_center', 'sum_xyz']], X_poly_df], axis=1)

# MinMaxScaler for X
scaler_X = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_X.fit_transform(X)

# Log transformation
y = np.log(y)

# MinMaxScaler for y
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.to_numpy().reshape(-1, 1))

# Split the data
X_scaled_train, X_scaled_test, y_scaled_train, y_scaled_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Indices of columns to be deleted
# This is just to fix a 'bug' that I have not identified
columns_to_delete = [5, 6, 7]

# Delete columns from X_scaled_train
X_scaled_train = np.delete(X_scaled_train, columns_to_delete, axis=1)

# Delete columns from X_scaled_test
X_scaled_test = np.delete(X_scaled_test, columns_to_delete, axis=1)
X_scaled = np.delete(X_scaled, columns_to_delete, axis=1)
X = X_scaled

# Print the updated shapes
print("Updated X_scaled_train shape:", X_scaled_train.shape)
print("Updated X_scaled_test shape:", X_scaled_test.shape)
#%% Toy/basic model of GP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Define the best kernel parameters
best_kernel_params = {'kernel__length_scale': 1.2927200214928352, 'kernel__nu': 1.540932921169848}

# Create the model pipeline with the best kernel parameters
kernel = Matern(length_scale=best_kernel_params['kernel__length_scale'], nu=best_kernel_params['kernel__nu'])
model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=10))

# Fit the model
model.fit(X_scaled_train, y_scaled_train)

# Make predictions on the test set
y_pred = model.predict(X_scaled_test)

# Output y_test and y_pred
print("True Labels (y_test): ", y_scaled_test)
y_test = y_scaled_test
print("Predicted Labels (y_pred): ", y_pred)

# Make predictions
y_pred, sigma = model.predict(X_scaled_test, return_std=True)

plt.figure()
plt.plot(range(0, len(y_scaled_test), 3), y_scaled_test[::3], "r.", markersize=10, label="Observations")
plt.plot(range(0, len(y_scaled_test), 3), y_pred[::3], "b-", label="Prediction")
plt.fill_between(
    range(0, len(y_scaled_test), 3),
    (y_pred - 1.9600 * sigma).flatten()[::3],
    (y_pred + 1.9600 * sigma).flatten()[::3],
    alpha=.5,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
y_test = y_scaled_test
plt.xlabel("Index")
plt.ylabel("Target (Scaled)")
plt.legend(loc="best")
plt.show()
#%%Feature Enginnering: Feature Attribution
print(model)
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance

# Permutation Importance
perm_importance = permutation_importance(model, X_scaled, y)
perm_importance_scores = perm_importance.importances_mean
print("Permutation Importance Scores:", perm_importance_scores)

features = ['x', 'y', 'z', 'distance_to_center', 'sum_xyz', 'x^2', 'y^2', 'z^2', 'xy', 'xz', 'yz']

# Permutation Importance
plt.bar(features, perm_importance_scores)
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.title('Permutation Importance')
plt.show() 
#%%Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression

# Correlation-based Feature Selection
correlation_scores = np.corrcoef(X_scaled.T, y_scaled.T)
correlation_scores = correlation_scores[:-1, -1]  # Exclude the target variable correlation
absolute_scores = np.abs(correlation_scores)
selected_features_corr = np.argsort(absolute_scores)[::-1]

# SelectKBest Feature Selection with F-test
skb = SelectKBest(score_func=f_regression, k=5)  # Select top 5 features based on F-test
X_skb_selected = skb.fit_transform(X_scaled, y_scaled)

# SelectKBest Feature Selection with Mutual Information
skb_mi = SelectKBest(score_func=mutual_info_regression, k=5)  # Select top 5 features based on mutual information
X_skb_mi_selected = skb_mi.fit_transform(X_scaled, y_scaled)

# Print the results
print("Correlation-based Feature Selection:")
print("Selected Features:", selected_features_corr)
print("Feature Importance Scores:", correlation_scores)
print("")

print("SelectKBest Feature Selection with F-test:")
print("Selected Features:", np.arange(X_scaled.shape[1])[skb.get_support()])
print("Feature Importance Scores:", skb.scores_)
print("")

print("SelectKBest Feature Selection with Mutual Information:")
print("Selected Features:", np.arange(X_scaled.shape[1])[skb_mi.get_support()])
print("Feature Importance Scores:", skb_mi.scores_)
#%%Select feature based on the result
selected_columns = [0, 3, 5, 9, 10]

X_scaled_test = X_scaled_test[:, selected_columns]
X_scaled_train = X_scaled_train[:, selected_columns]

#I directly uniform the variables, should split variables and be more clear next time!
X_train = X_scaled_train
X_test = X_scaled_test
X_val = X_scaled_test
X_scaled = X_scaled[:, selected_columns]
X = X[:, selected_columns]

y_val = y_scaled_test
y_test = y_scaled_test

#%%Bayesian method for optimizing parameters
import optuna
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ignore the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to optimize
    kernel = trial.suggest_categorical("kernel", ["Matern"])
    alpha = trial.suggest_float("alpha", 0.1, 2.0)

    if kernel == "Matern":
        nu = trial.suggest_float("nu", 0.1, 2.0)
        length_scale = trial.suggest_float("length_scale", 0.1, 2.0)
        kernel = Matern(length_scale=length_scale, nu=nu)

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

    # Fit the data
    gpr.fit(X_scaled_train, y_scaled_train)

    # Calculate the mean squared error on the validation set
    y_pred = gpr.predict(X_scaled_test)
    mse = ((y_pred - y_scaled_test) ** 2).mean()

    return mse

# Split the data into train and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X)
y_scaled_train = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# Create the Optuna study
study = optuna.create_study(direction="minimize")

# Optimize the objective function
study.optimize(objective, n_trials=20)

# Get the best hyperparameters and create the final GaussianProcessRegressor
best_params = study.best_params
kernel = best_params["kernel"]
alpha = best_params["alpha"]

if kernel == "Matern":
    nu = best_params["nu"]
    length_scale = best_params["length_scale"]
    kernel = Matern(length_scale=length_scale, nu=nu)

gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
gpr.fit(X_scaled_train, y_scaled_train)
#[I 2023-08-29 11:44:40,940] Trial 16 finished with value: 0.2522736767461608 and parameters: {'kernel': 'Matern', 'alpha': 1.816039649753506, 'nu': 0.6747152160634156, 'length_scale': 1.9665926236344484}. Best is trial 16 with value: 0.2522736767461608.

# I have improved the model using Feature Enginnering and Bayesian Optimization 
#%% Improved Gaussian process based on privious improvement
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Define the best kernel parameters
best_kernel_params = {'kernel__length_scale': 1.965, 'kernel__nu': 0.67}

# Create the model pipeline with the best kernel parameters
kernel = Matern(length_scale=best_kernel_params['kernel__length_scale'], nu=best_kernel_params['kernel__nu'])
model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=1.87, n_restarts_optimizer=10))

# Fit the model
model.fit(X_scaled_train, y_scaled_train)

# Make predictions
y_pred, sigma = model.predict(X_scaled_test, return_std=True)

plt.figure()
plt.plot(range(0, len(y_scaled_test), 3), y_scaled_test[::3], "r.", markersize=10, label="Observations")
plt.plot(range(0, len(y_scaled_test), 3), y_pred[::3], "b-", label="Prediction")
plt.fill_between(
    range(0, len(y_scaled_test), 3),
    (y_pred - 1.9600 * sigma).flatten()[::3],
    (y_pred + 1.9600 * sigma).flatten()[::3],
    alpha=.5,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
y_test = y_scaled_test
X_test = X_scaled_test
plt.xlabel("Index")
plt.ylabel("Target (Scaled)")
plt.legend(loc="best")
plt.show()
#%% Test overfitting
from sklearn.model_selection import train_test_split

# Evaluate performance on the training set
y_pred_train = model.predict(X_scaled_train)
mse_train = mean_squared_error(y_pred_train, y_scaled_train)
print("MSE on training set:", mse_train)

# Evaluate performance on the validation set
mse_val = mean_squared_error(y_test, y_pred)
print("MSE on validation set:", mse_val)

plt.scatter(y_pred_train, y_scaled_train, label="Training Data")
plt.scatter(y_test, y_pred, label="Validation Data")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()
# Model still counter severe overfitting
# From the learning curve below, I found data is not enough for the model to study all patterns
#%% Data augmentation 
# Simple data augmentation method
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import make_pipeline

# Add a smaller amount of noise
noise_std = 0.1  # Smaller noise standard deviation
noise = np.random.normal(0, noise_std, size=(X_scaled_train.shape[0], X_scaled_train.shape[1]))  # Generate noise with correct shape

additional_samples_ratio = 0.7  # Additional samples ratio
num_additional_samples = int(additional_samples_ratio * X_scaled_train.shape[0])  # Number of additional samples

# Randomly select indices for additional samples
additional_indices = np.random.choice(X_scaled_train.shape[0], size=num_additional_samples, replace=False)

# Apply noise to X_scaled_train for additional samples
X_scaled_additional = X_scaled_train[additional_indices] + noise[additional_indices]

# Apply noise to y_scaled_train for additional samples
y_additional = y_scaled_train[additional_indices].reshape(-1, 1) + noise[additional_indices][:, 0].reshape(-1, 1)

# Concatenate additional samples with original data
X_scaled_augmented = np.vstack((X_scaled_train, X_scaled_additional))
y_augmented = np.vstack((y_scaled_train.reshape(-1, 1), y_additional))

# Define the best kernel parameters
best_kernel_params = {'kernel__length_scale': 1.965, 'kernel__nu': 0.67}

# Create the model pipeline with the best kernel parameters
kernel = Matern(length_scale=best_kernel_params['kernel__length_scale'], nu=best_kernel_params['kernel__nu'])
model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10))

# Fit the model on the augmented data
model.fit(X_scaled_augmented, y_augmented)

# Make predictions on the original test set
y_pred = model.predict(X_scaled_test)

# Output y_test and y_pred
print("True Labels (y_test): ", y_scaled_test)
print("Predicted Labels (y_pred): ", y_pred)

#All code below are exploratory
#%% Data augmentation 
# Augmentation using SMOTE (codes to 655 might include problem)
# Class Smote 
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

#%% Augmentation using SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
# Initialize SMOTE with the desired sampling strategy
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Perform SMOTE oversampling
X_resampled, y_resampled = smote.fit_resample(X, y)

# plot the augmented data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', label='Original Data')
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, marker='x', label='Resampled Data')
plt.legend()
plt.title('SMOTE Over-sampling')
plt.show()
#%%Rational Quadratic best: size = 0.1,seed = 20, alpha = 1, length_scale = 1
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
#%%Take the augmentation in the test set
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
#%% Improved Gaussian process based on privious improvement
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Define the best kernel parameters
best_kernel_params = {'kernel__length_scale': 1.965, 'kernel__nu': 0.67}

# Create the model pipeline with the best kernel parameters
kernel = Matern(length_scale=best_kernel_params['kernel__length_scale'], nu=best_kernel_params['kernel__nu'])
model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=1.87, n_restarts_optimizer=10))

# Fit the model
model.fit(X_scaled_train, y_scaled_train)

# Make predictions
y_pred, sigma = model.predict(X_scaled_test, return_std=True)

plt.figure()
plt.plot(range(0, len(y_scaled_test), 3), y_scaled_test[::3], "r.", markersize=10, label="Observations")
plt.plot(range(0, len(y_scaled_test), 3), y_pred[::3], "b-", label="Prediction")
plt.fill_between(
    range(0, len(y_scaled_test), 3),
    (y_pred - 1.9600 * sigma).flatten()[::3],
    (y_pred + 1.9600 * sigma).flatten()[::3],
    alpha=.5,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
y_test = y_scaled_test
X_test = X_scaled_test
plt.xlabel("Index")
plt.ylabel("Target (Scaled)")
plt.legend(loc="best")
plt.show()
#%%Further Explore: manifold + GP
# Import the required libraries
from sklearn.gaussian_process.kernels import Matern
from sklearn.manifold import Isomap

# Define the best kernel parameters
best_kernel_params = {'kernel__length_scale': 1.965, 'kernel__nu': 0.67}

# Define the best manifold dimension
best_manifold_dim = 2  # Adjust according to your needs

# Create an Isomap object for manifold learning
isomap = Isomap(n_components=best_manifold_dim)

# Perform manifold learning on the training set and map the test set to the same low-dimensional space
X_low_train = isomap.fit_transform(X_scaled_train)
X_low_test = isomap.transform(X_scaled_test)

# Create a Gaussian Process Regression object using the Matern kernel
kernel = Matern(length_scale=best_kernel_params['kernel__length_scale'], nu=best_kernel_params['kernel__nu'])
model = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)

# Fit the Gaussian Process Regression model in the low-dimensional space
model.fit(X_low_train, y_scaled_train)

# Make predictions in the low-dimensional space and transform the results back to the original space
y_pred, sigma = model.predict(X_low_test, return_std=True)
y_pred = scaler_y.inverse_transform(y_pred)
sigma = scaler_y.inverse_transform(sigma)
#%%Imitating an article：Leveraging Uncertainty in Machine Learning Accelerates Biological Discovery and Design
#MLP + GP algorithm (In Scikit-learn's MLPRegressor, there is no direct method for dimensionality reduction.)
# Import the required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.neural_network import MLPRegressor

# Define the best kernel parameters
best_kernel_params = {'kernel__length_scale': 1.965, 'kernel__nu': 0.67}

# Define the best MLP parameters
best_mlp_params = {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'alpha': 0.01}

# Create an MLPRegressor object using the best parameters
mlp = MLPRegressor(hidden_layer_sizes=best_mlp_params['hidden_layer_sizes'], activation=best_mlp_params['activation'], alpha=best_mlp_params['alpha'])

# Fit the MLPRegressor model on the training set and transform the test set to the same low-dimensional space
X_low_train = mlp.transform(X_scaled_train)
X_low_test = mlp.transform(X_scaled_test)

# Create a Gaussian Process Regression object using the Matern kernel
kernel = Matern(length_scale=best_kernel_params['kernel__length_scale'], nu=best_kernel_params['kernel__nu'])
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)

# Fit the Gaussian Process Regression model in the low-dimensional space
gpr.fit(X_low_train, y_scaled_train)

# Make predictions in the low-dimensional space and transform the results back to the original space
y_pred, sigma = gpr.predict(X_low_test, return_std=True)
y_pred = scaler_y.inverse_transform(y_pred)
sigma = scaler_y.inverse_transform(sigma)

#%% PCA + GP
from sklearn.decomposition import PCA

# Create a PCA object and specify the dimensionality after reduction
pca = PCA(n_components=2)

# Fit the PCA model on the training set and transform both the training and test sets to the low-dimensional space
X_low_train = pca.fit_transform(X_scaled_train)
X_low_test = pca.transform(X_scaled_test)

# Create a Gaussian Process Regression object using the Matern kernel
kernel = Matern(length_scale=best_kernel_params['kernel__length_scale'], nu=best_kernel_params['kernel__nu'])
model = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)

# Fit the Gaussian Process Regression model in the low-dimensional space
model.fit(X_low_train, y_scaled_train)

# Make predictions in the low-dimensional space and transform the results back to the original space
y_pred, sigma = model.predict(X_low_test, return_std=True)
y_pred = scaler_y.inverse_transform(y_pred)
sigma = scaler_y.inverse_transform(sigma)

#%% Ensemble learning
#%% Neural Network
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Create the neural network model
model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), activation='tanh', solver='adam', alpha=0.01, random_state=42)

# Model training
model.fit(X_scaled_train, y_scaled_train)

# Make predictions on the test set
y_pred = model.predict(X_scaled_test)
print(y_pred)

#%% Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Create the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Model training
model.fit(X_scaled_train, y_scaled_train)

# Make predictions on the test set
y_pred = model.predict(X_scaled_test)
print(y_pred)

#%% SVM
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Define the parameter space
param_grid = {
    'C': [3, 4, 10, 15],
    'epsilon': [0.5, 0.8, 1, 2]
}

# Create the SVR model
model = SVR(kernel='rbf')

# Create the grid search object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the model and perform parameter search on the training set
grid_search.fit(X_scaled_train, y_scaled_train)

# Output the best parameter combination
print("Best parameters: ", grid_search.best_params_)

# Make predictions using the model with the best parameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_scaled_test)

print(y_pred)

#%% Ensemble Learning with the four models above
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the base learners
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
nn_model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), activation='tanh', solver='adam', alpha=0.01, random_state=42)
svm_model = SVR(C=10, epsilon=1)
best_kernel_params = {'kernel__length_scale': 1.2927200214928352, 'kernel__nu': 1.540932921169848}
kernel = Matern(length_scale=best_kernel_params['kernel__length_scale'], nu=best_kernel_params['kernel__nu'])
gp_model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=10))

# Train the base learners
rf_model.fit(X_scaled_train, y_scaled_train)
nn_model.fit(X_scaled_train, y_scaled_train)
svm_model.fit(X_scaled_train, y_scaled_train)
gp_model.fit(X_scaled_train, y_scaled_train)

# Use base learners' predictions as new features
rf_predictions = rf_model.predict(X_scaled_train)
nn_predictions = nn_model.predict(X_scaled_train)
svm_predictions = svm_model.predict(X_scaled_train)
gp_predictions = gp_model.predict(X_scaled_train)

# Build the new feature matrix
stacking_X_train = np.column_stack((rf_predictions, nn_predictions, svm_predictions, gp_predictions))

# Build the meta learner
meta_model = LinearRegression()
meta_model.fit(stacking_X_train, y_scaled_train)

# Use base learners' predictions on the test set as new features
rf_test_predictions = rf_model.predict(X_scaled_test)
nn_test_predictions = nn_model.predict(X_scaled_test)
svm_test_predictions = svm_model.predict(X_scaled_test)
gp_test_predictions = gp_model.predict(X_scaled_test)

# Build the new feature matrix
stacking_X_test = np.column_stack((rf_test_predictions, nn_test_predictions, svm_test_predictions, gp_test_predictions))

# Use the meta learner for predictions
stacking_predictions = meta_model.predict(stacking_X_test)
y_pred = stacking_predictions
print(y_pred)

#%%Evaluation basic
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# Calculate mean value of y_test
y_mean = np.mean(y_test)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate total sum of squares (TSS)
tss = np.sum((y_test - y_mean)**2)

# Calculate regression sum of squares (RSS)
rss = np.sum((y_test - y_pred)**2)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate coefficient of determination (R2)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Total Sum of Squares (TSS): {tss}")
print(f"Regression Sum of Squares (RSS): {rss}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

#%% Evaluation for CI
# Calculate confidence interval for index
confidence_level = 0.95  # Confidence level of 95%
error = y_test - y_pred
mean_error = np.mean(error)
std_error = np.std(error)
z_score = 1.96  # Z-value for 95% confidence interval of normal distribution
lower_bound = y_pred - z_score * std_error
upper_bound = y_pred + z_score * std_error

# Reduce the number of data points to display
step = 10  # Display every 10th data point
indices = np.arange(0, len(y_test), step)

# Adjust the figure size
plt.figure(figsize=(12, 6))

# Plot scatter plot and confidence interval
plt.scatter(indices, y_test[indices], color='red', label='Actual')
plt.scatter(indices, y_pred[indices], color='blue', label='Predicted')
plt.fill_between(indices, lower_bound[indices].squeeze(), upper_bound[indices].squeeze(), color='lightblue', alpha=0.5, label='Confidence Interval')

plt.xlabel('Data Point')
plt.ylabel('y')
plt.title('Prediction with Confidence Interval')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate confidence interval in 3D
confidence_level = 0.95  # Confidence level of 95%
error = y_test - y_pred
mean_error = np.mean(error)
std_error = np.std(error)
z_score = 1.96  # Z-value for 95% confidence interval of normal distribution
lower_bound = y_pred - z_score * std_error
upper_bound = y_pred + z_score * std_error

# Reduce the number of data points to display
step = 1  # Display every 1st data point
indices = np.arange(0, len(y_test), step)

# Adjust the figure size
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot scatter plot and confidence interval
ax.scatter(X_test[indices, 0], X_test[indices, 1], y_test[indices], color='red', label='Actual')
ax.scatter(X_test[indices, 0], X_test[indices, 1], y_pred[indices], color='blue', label='Predicted')
ax.plot_trisurf(X_test[indices, 0], X_test[indices, 1], lower_bound[indices].squeeze(), color='lightblue', alpha=0.5)
ax.plot_trisurf(X_test[indices, 0], X_test[indices, 1], upper_bound[indices].squeeze(), color='lightblue', alpha=0.5)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_title('Prediction with Confidence Interval')
ax.legend()
plt.tight_layout()
plt.show()
#%% Evaluation the properties of CI
#1.Coverage
def coverage(y_true, y_pred, y_std, alpha):
    lower_bound = y_pred - y_std * np.sqrt(1 / alpha)
    upper_bound = y_pred + y_std * np.sqrt(1 / alpha)
    covered = np.logical_and(lower_bound <= y_true, y_true <= upper_bound)
    return np.mean(covered)

coverage_score = coverage(y_test, y_pred, std_error, alpha)
print("Coverage: ", coverage_score)

#2.Average Width
def average_width(y_std):
    return np.mean(y_std)

width_score = average_width(std_error)
print("Average Width: ", width_score)

#3.Average Length
def average_length(y_std, y_true_range):
    return np.mean(y_std) / y_true_range

length_score = average_length(std_error, y_test.max() - y_test.min())
print("Average Length: ", length_score)

#4.prediction variance
def prediction_variance(y_std):
    return np.mean(y_std**2)

variance_score = prediction_variance(std_error)
print("Prediction Variance: ", variance_score)
#%%Evaluation: Learning curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
# Define a function to plot the learning curve
def plot_learning_curve(estimator, X, y, train_sizes, cv):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv, scoring='neg_mean_squared_error')

    # Calculate the average training scores, validation scores, and standard deviations
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    plt.xlabel('Training Set Size')
    plt.ylabel('Negative Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Define the proportions of training set sizes
train_sizes = np.linspace(0.1, 1.0, 5)

# Plot the learning curve
plot_learning_curve(model, X_scaled_augmented, y_augmented, train_sizes, cv=5)
#%% Evaluation: distance between resides
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Assuming x, y, z are already defined with the data
points = np.column_stack((x, y, z))

# Build a KDTree for efficient nearest neighbor search
kdtree = cKDTree(points)

# Find the 50 nearest neighbors for each point
distances, indices = kdtree.query(points, k=51)  # k=51 to include the point itself

# Exclude the first column of indices, which corresponds to the point itself
nearest_indices = indices[:, 1:]

# Get the distances to the nearest neighbors
nearest_distances = distances[:, 1:]

# Create a DataFrame with the distances
df = pd.DataFrame(nearest_distances, columns=[f"Neighbor_{i}" for i in range(1, 51)])

# Plot a histogram for each column, in batches of 10
num_batches = 5
batch_size = 10

for i in range(num_batches):
    start_index = i * batch_size
    end_index = (i + 1) * batch_size

    batch_df = df.iloc[:, start_index:end_index]

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()

    for j, column in enumerate(batch_df.columns):
        axs[j].hist(batch_df[column], bins=10)
        axs[j].set_title(column)

    plt.tight_layout()
    plt.show()

#%% Evaluation: Likelihood (has not finished yet)
LML = [
    [
        gp_opt.log_marginal_likelihood(np.log([Theta0[i, j],
                                               Theta1[i, j]]))
        for i in range(Theta0.shape[0])
    ]
    for j in range(Theta0.shape[1])
]
LML = np.array(LML).T
plt.contour(Theta0, Theta1, LML)
plt.scatter(
    gp_opt.kernel_.theta[0],
    gp_opt.kernel_.theta[1],
    c="r",
    s=50,
    zorder=10,
    edgecolors=(0, 0, 0),
)
plt.plot(
    [gp_fix.kernel_.theta[0]],
    [gp_fix.kernel_.theta[1]],
    "bo",
    ms=10,
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Length scale")
plt.ylabel("Noise level")
plt.title("Log-marginal-likelihood")



# Plot LML as a function of length scale
plt.figure()
plt.plot(
    gp_opt.kernel_.theta[0],
    gp_opt.log_marginal_likelihood(gp_opt.kernel_.theta),
    "bo",
    ms=10,
)
plt.plot(
    gp_fix.kernel_.theta[0],
    gp_fix.log_marginal_likelihood(gp_fix.kernel_.theta),
    "ro",
    ms=10,
)
plt.xlabel("Length scale")
plt.ylabel("Log-marginal-likelihood")
plt.title("Log-marginal-likelihood as\
a function of length scale")
  
plt.show()


 


