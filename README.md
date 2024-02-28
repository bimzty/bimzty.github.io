This webpage stores the projects that I have done during my undergraduate
## About my self:
<p align="center">
  <img src="https://github.com/bimzty/bimzty.github.io/blob/main/Photos/MY%20PHOTO%20II.png" width="400" />
</p>
CV: https://maipdf.com/est/a14080205457@pdf <br>
Certification: https://maipdf.com/est/d17353768550@pdf<br>
My photos: https://maifile.cn/est/a65ba089c55e0a/pdf <br>

## Underway: 
I am evolving in the research aimed at 'Addressing Class Imbalance Issues in Financial Anti-Fraud Detection with Data Preprocessing and Machine Learning'.<br>
Specifically, I am now comparing different data augmentation method including GAN and SMOTE, and analyze those algorithms from principle. <br>
Tutor：Boon Giin Lee https://research.nottingham.edu.cn/en/persons/boon-giin-lee<br>
This research is expected to publish a research article

***

## 1.Research for Spike protein on SARS-CoV-2 virus (07/2023-08/2023)

Position: Data Analyst Intern <br>
Location: Shenzhen Bay Laboratory. <br>
tutor: Chaowang https://www.szbl.ac.cn/en/scientificresearch/researchteam/3372.html

I was mainly involved in identifying mutational hotspots and capturing the mutation distribution using the Gaussian process on the SARS-CoV-2 spike protein.

### Background & Explain
Mutational Hotspots are the places on the protein where the mutation frequency is higher than in other places, they are the main consideration for designing experiments and vaccines. Furthermore, we would like to capture the regression patterns for this protein, for potential prediction tasks in future research.

### 1.1 Applying a weighted average proximity scoring function for identifying hotspots
Check the 'Identifying hotspots use WAP method' Rmd file. The idea of the method is from a Comprehensive assessment of cancer missense mutation clustering in protein structures by Atanas Kamburov et, al. The WAP algorithm from this article is first used for identifying clusters that are significant for mutation. I furthermore improve the methods by adding an adaptive step for finding the optimal size of each cluster before the original process.

### 1.2 Applying K-means or DBSCAN method for identifying hotspots
Check the 'Identifying hotspots using Clustering method' R file for finding hotspots. Firstly, this method performs Data Preprocessing to transform the spatial position('X',' Y',' Z') of residuals and VirusPercentage to the same scale<br> Secondly, this method examines the data distribution invariant before and after the process. <br>Thirdly, the method uses K-means to cluster different residues. <br>Fourthly, it performs the Permutation methods to examine the significance of mutation frequency for different clusters. <br>Various hypothesis tests have been conducted in this step. We finally utilize T-SNE for dimensionality reduction and visualization.

### 1.3 Applying the Gaussian process for capturing the mutational distribution of the protein.
Check the 'Gaussian process for mutation distribution'. Data processing here includes Log transformation and Box-Cox transformation for 'Mutation Number'. I first implemented the basic GP method with the kernel function using Matern, a generalization of RBF, and using Random search for the parameters in the kernel function. This basic model gives a tragic result. 

I thus improved the model by following methods:<br>
#### 1.3.1. Feature Engineering: 
I extracted more features from the original spatial positions ('X',' Y',' Z') and selected them. <br>Feature extraction: from ('X',' Y',' Z') to (x',' y', 'z', 'distance_to_center', 'sum_xyz', 'x^2', 'y^2', 'z^2', 'xy', 'xz', 'yz').<br> Feature attribution: SHAP (Shapley Additive explanations) and Permutation Importance measures are implemented.<br> Feature selection: Correlation-based method and SelectKBest Feature Selection were applied. I finally chose  ‘x’,’ y’, 'x^2',’ xy,’xz’, and ‘‘distance to the center’ as my final features.<br>


   
#### 1.3.2. The Bayesian optimization:
The Bayesian optimization method was also learned and replaced with the Random search method. The Bayesian process for Gaussian Process kernel selection involves iteratively evaluating and updating kernel configurations based on prior beliefs and observed data. It efficiently explores the search space, exploits prior knowledge, and provides uncertainty estimates. This is better than random search because it intelligently guides the search towards promising regions, utilizes past information, builds a surrogate model, and converges to better solutions faster.
 
#### 1.3.3. Data Augmentation: 
As generally the residues with high mutation numbers are in the minority, there is an obvious imbalanced distribution of mutations in the dataset. I applied SMOTE (Synthetic Minority Over-sampling Technique) to solve the problem. The detailed program is at 'SMOTE_resample.py', the program could create synthetic samples that lie on the line segments between existing minority class samples, and helps to increase the representation of the minority class and reduce the imbalance in the dataset. I utilized the learning curves show this approach can effectively improve the result.


<p align="center">
  <img src="https://github.com/bimzty/bimzty.github.io/blob/main/Photos/GP%20without%20Augmentation.png" width="400" />
  <img src="https://github.com/bimzty/bimzty.github.io/blob/main/Photos/GP%20woth%20Augmentation.png" width="400" />
</p>
<p align="center">
  <em>The difference of regression value and real value when without augmentation (left) and with augmentation</em>
</p>

#### 1.3.4 The original model has an overfitting problem. 
Best Parameterset for Regularization have been searched and applied. I have approached this by two methods: <br>
1. Selecting a simpler kernel function or reducing the number of hyperparameters <br>
2. Use Bayesian inference to estimate the posterior distribution over the model parameters.
  
***

## 2.Machine Learning Research: Classification task for Freddie Mac loan dataset and Historical Stock Market dataset(05/2022-08/2022, 09/2023)
Location: University of Nottingham, Ningbo, china.
Tutor: Saeid Pourroostaei Ardakani  https://scholar.google.com/citations?user=3OeHr8gAAAAJ

I was mainly involved in delivering literature research related to Federated learning, implemented Feature Engineering, and built and compared multiple models in both tasks.

### 2.1 Classification task for Freddie Mac loan dataset 
This research is conducted in a team, and I only demonstrate the program I wrote. Check 2.1 ’Report of Result’ for several results I conducted utilizing models built by myself or other team members.

### 2.2 Historical Stock Market Dataset
Check "2.2 ML research for Predicting Stock Market.py". <br>Data preprocessing: Transformation, cross-sectional standardization <br> Feature Engineering: Feature Extraction: from a financial perspective (Bollinger Bands and waveform-based methods) <br> Feature Attribution： Saliency, Integrated Gradients, and Shapely Value Sampling Methods <br> Feature Selection: Static and dynamic Feature Selection Self-Attention Mechanism<br> Model: LSTM and GRU; Also tried: Linear Regression; Decision Tree; Random Forest; <br> Explore: 1. Add Self-Attention Mechanism in LSTM and GRU model to improve long term memory; <br> 2.Visualize features from Neural Networks. (using TSNE)

<p align="center">
  <img src="https://github.com/bimzty/bimzty.github.io/blob/main/Photos/LSTM%20for%20predicting%20stock%20price.png " width="400" />
</p>
<p align="center">
  <em>Price Prediction using LSTM model</em>
</p>

Check 'study attention mechanism from Google article' for relevant code I produced when learning the article 'Attention Is All You Need' by Ashish Vaswani et, al.

This year, out of interest in Time Series and Quantitive Finance, I have systematically written a more systematic and true-to-life project for this task, which you can see in the folder 'A systematic approach for Quantitive Trading'. Instead of using a neural network, I emphasized the Support Vector Machine this time after being immersed in three related articles I included in that folder. I also utilized MovingAverageCrossStrategy (in '4_mac') for conducting strategy with events in the market being considered.

***

## 3. Internship: Mathematical modeling intern at ZHONGCE RUBBER GROUP CO., LTD.(06/2022-07/2022)

Worked as a Mathematical Modelling intern
Design a program for the Hans B. Pacejka empirical tire model based on experimental data.

Hans B. Pacejka model: https://en.wikipedia.org/wiki/Hans_B._Pacejka
 <br>
The program code: 1. calculate the parameters for the model<br>
2. Draw a series of analytical graphs

<p align="center">
  <img src="https://github.com/bimzty/bimzty.github.io/blob/main/Photos/Tire%20Dynamic%20Analysis%20Lateral%20force%20versus%20slip%20angle..png" width="300" />
</p>
<p align="center">
 <em>Tire Dynamic Analysis：Tire lateral force versus slip angle.</em>
</p>

You can check the 'H.B.Pacejka model_Sample_GUI' MATLAB Figure program alongside the MATLAB GUI of the program. 
***
## 4. Competition: Formula Student Electric, won National Third Prize (12/2020-07/2022)
Responsibility: Data Analyst and Simulation Technician<br>
The 'Track Simulation' folder consists of Python codes mainly developed to stimulate racecar performance on Track. It consists of functions for: <br>
Straight road, when the car accelerates only;<br>
Corner, when the car runs the maximum speed under the condition;<br>
Brake, when the car decelerates only;
<p align="center">
  <img src=" https://github.com/bimzty/bimzty.github.io/blob/main/Photos/Track%20Simulations.png " width="400" />
</p>

***

## 5. Coursework during Undergraduate 
### 5.1 Coursework for the module Machine Learning (12/2023)
The coursework aims to make use of the machine learning techniques learned in this course to diagnose breast cancer using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Based on the recommended model and parameters of a similar competition held by Kaggle, I mainly built 7 models and conducted a systematic approach to choosing the best one from them. 

### 5.2 Coursework for the module Introduction to Scientific Computation (09/2022-05/2023)
This course aims to introduce the concept of numerical approximation to problems that cannot be solved analytically and to develop skills in Python by implementing numerical methods. Topics included in those works are: Solving nonlinear equations (approximately) using root finding methods and analyzing their convergence; Solving linear systems of equations using direct methods and iterative techniques, including Gaussian elimination and Jacobi & Gauss-Seidel method; Approximating functions by polynomial interpolants (Lagrange polynomials), and analyzing their accuracy; Approximating derivatives and definite integrals using numerical differentiation and integration such as trapezoidal, Simpson & midpoint rule, and analyzing their convergence; Approximating ODEs using numerical methods including Euler’s method and s, higher-order RK methods.

### 5.3 Coursework for the module Statistical Models and Methods (05/2023))
The objective of the coursework is to build a predictive model for body fat content using 10 body measurement variables. we first do some exploratory analysis of the data. Secondly, we do model selection to find the best subset of variables for regression based on AIC/BIC, Mallow’s Cp, and Adjusted
R-squared criterion. Thirdly, we identify and analyze outliers and high-leverage points. Fourthly, we check the linear
model assumption by plotting the QQ-plot, residual, component residual plot, etc, and do the manipulation to a model
based on that. A comparison between our best model and the full model using test data will also be provided.

















