This webpage stores the projects that I have done during my undergraduate
## About my self:
CV: https://maipdf.com/est/a14080205457@pdf
Academic Transcript (please note that my transcript has been updated in 12/2023): https://maipdf.com/est/a15151202453@pdf
Certificate:
Some photos:

## Underway: 
I am evolving in the research aimed at 'Addressing Class Imbalance Issues in Financial Anti-Fraud Detection with Data Preprocessing and Machine Learning'.
Tutor：Boon Giin Lee https://research.nottingham.edu.cn/en/persons/boon-giin-lee
This research is expected to publish a research article


## 1.Research for Spike protein on SARS-CoV-2 virus (07/2023-08/2023)

Position: Data Analyst Intern
Location: Shenzhen Bay Laboratory.
tutor: Chaowang https://www.szbl.ac.cn/en/scientificresearch/researchteam/3372.html

I was mainly involved in identifying mutational hotspots and capturing the mutation distribution using the Gaussian Process on the SARS-CoV-2 spike protein.

### Background & Explain
Mutational Hotspots are the places on the protein where the mutation frequency is higher than in other places, they are the main consideration for designing experiments and vaccines. Furthermore, we would like to capture the regression patterns for this protein, for potential prediction tasks in future research.

### 1.1 Applying a weighted average proximity scoring function for identifying hotspots
Check the 'Identifying hotspots use WAP method' Rmd file.

### 1.2 Applying K-means or DBSCAN method for identifying hotspots
Check the 'Identifying hotspots using Clustering method' R file for finding hotspots. Firstly, this method performs Data Preprocessing to transform the spatial position('X',' Y',' Z') of residuals and VirusPercentage to the same scale[add more??????]. Secondly, this method examines the data distribution invariant before and after the process. Thirdly, the method uses K-means to cluster different residues. Fourthly, I perform the Permutation methods to examine the significance of mutation frequency for different clusters. Various hypothesis tests have been conducted in this step. We finally utilize T-SNE for dimensionality reduction and visualization.

### 1.3 Applying the Gaussian process for capturing the mutational distribution of the protein.
Check the 'Gaussian Process for mutation distribution'. Data processing here includes log transformation and Box-Cox transformation for 'Mutation Number'. I first implemented the basic GP method with the kernel function using Matern, a generalization of RBF, and using Random Search for the parameters in the kernel function. This basic model gives a tragic result. 

I thus improved the model by following methods:
1.3.1. Feature Engineering: I extracted more features from the original spatial positions ('X','Y','Z') and selected them. Feature extraction: from ('X','Y','Z') to (x','y','z','distance_to_center', 'sum_xyz','x^2', 'y^2', 'z^2', 'xy', 'xz', 'yz'). Feature attribution: SHAP (SHapley Additive explanations) and Permutation Importance measures are implemented. Feature selection: Correlation-based method and SelectKBest Feature Selection were applied. I finally chose  ‘x’,’y’,'x^2',’ xy,,’xz’ and ‘‘distance to the center’ as my final features.
   
1.3.2. The Bayesian optimization method was also learned and replaced with the Random search method. (more detail being add...)
 
1.3.3. Data Augmentation: As there is an obvious imbalanced distribution of mutations in the dataset, I applied SMOTE to solve the problem. The detailed program is at 'SMOTE for mutation distribution of the SPIKE protein '. I tried three approaches. No augmentation at all; 2. Augmentation on training set; 3. Augmentation on the entire set. We can see from the learning curve [picpicpicpic....] that only when augmenting on the entire dataset can we have dramatic improvement.
  
1.3.4. Regularization (more detail being add...)

## 2.Machine Learning Research: Classification task for Freddie Mac loan dataset and Historical Stock Market dataset(05/2022-08/2022, 09/2023)
Location: University of Nottingham, Ningbo, china.
Tutor: Saeid Pourroostaei Ardakani  https://scholar.google.com/citations?user=3OeHr8gAAAAJ

I was mainly involved in delivering literature research related to Federated learning, implemented Feature Engineering, and built and compared multiple models in both tasks.

### 2.1 Classification task for Freddie Mac loan dataset 
This research is conducted in a team, and I only demonstrate the program I wrote. Check 2.1 ’Report of Result’ for several results I conducted utilizing models built by myself or other team members.

### 2.2 Historical Stock Market Dataset
Data preprocessing: Transformation, cross-sectional standardization <br> Feature Engineering: Feature Extraction: from a financial perspective (Bollinger Bands and waveform-based methods) <br> Feature Attribution： Saliency, Integrated Gradients, and Shapely Value Sampling Methods <br> Feature Selection: Static and dynamic Feature Selection Self-Attention Mechanism<br> Model: LSTM and GRU; Also tried: Linear Regression; Decision Tree; Random Forest; <br> Explore: 1. Add Self-Attention Mechanism in LSTM and GRU model to improve long term memory; 2.Finding features from Neural Networks (using TSNE)

Check 'study attention mechanism from Google article' for relevant code I produced when learning the article 'Attention Is All You Need' by Ashish Vaswani et, al.

This year, out of interest in Time Series and Quantitive Finance, I have systematically written a more systematic and true-to-life project for this task, which you can see in the folder 'A systematic approach for Quantitive Trading'. Instead of using a neural network, I emphasized the Support Vector Machine this time after being immersed in three related articles I included in that folder. I also utilized MovingAverageCrossStrategy (in '4_mac') for conducting strategy with events in the market being considered.

 ## 3. Internship: Mathematical modeling intern at ZHONGCE RUBBER GROUP CO., LTD.(06/2022-07/2022)

Worked as a Mathematical Modelling intern
Design a program for the Hans B. Pacejka model, which is an empirical tire model based on experimental data.

Hans B. Pacejka model: https://en.wikipedia.org/wiki/Hans_B._Pacejka

The program code: 1. calculate the parameters for the model
2. Draw a series of analytical graphs

You can check the 'H.B.Pacejka model_Sample_GUI' MATLAB Figure program alongside the MATLAB GUI of the program. 

## 4. Competition: Formula Student Electric, won National Third Prize (12/2020-07/2022)
Responsibility: Data Analyst and Simulation Technician

1. Developed functions in Python for simulating car performance on the track (check '')
2. Design genetic algorithms to optimize the design of chassis components (check '')

## 5. Coursework during Undergraduate 
### 5.1 Coursework for the module Machine Learning (12/2023)
The coursework aims to make use of the machine learning techniques learned in this course to diagnose breast cancer using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Based on the recommended model and parameters of a similar competition held by Kaggle, I mainly built 7 models and conducted a systematic approach to choosing the best one from them. 

### 5.2 Coursework for the module Introduction to Scientific Computation
This course aims to introduce the concept of numerical approximation to problems that cannot be solved analytically and to develop skills in Python through the implementation of numerical methods. Topics included in those works are: solving nonlinear equations (approximately) using root finding methods and analyzing their convergence; Solving linear systems of equations using direct methods, including Gaussian elimination; Solving linear systems of equations using iterative techniques, including Jacobi & Gauss-Seidel method; Approximating functions by polynomial interpolants (Lagrange polynomials), and analyzing their accuracy; Approximating derivatives and definite integrals using numerical differentiation and integration such as trapezoidal, Simpson & midpoint rule, and analyzing their convergence; Approximating ODEs using numerical methods including Euler’s method and s, higher-order RK methods.

















