This webpage stores the projects that I have done during my undergraduate
## About my self:
<p align="center">
  <img src="https://github.com/bimzty/bimzty.github.io/blob/main/Photos/MY%20PHOTO%20II.png" width="400" />
</p>

Certification: https://maipdf.com/est/d17353768550@pdf<br>
My photos: https://maifile.cn/est/d65df34e842643/pdf <be>

## Research underway: 
1. I am evolving in the research aimed at 'Addressing Class Imbalance Issues in Financial Anti-Fraud Detection with Data Preprocessing and Machine Learning'.<br>
Specifically, I am now comparing different data augmentation methods including GAN and SMOTE, and analyzing those algorithms in principle. <br>
Tutor：Boon Giin Lee (World top2% Scientist) https://research.nottingham.edu.cn/en/persons/boon-giin-lee<br>
This research is expected to publish a research article.

***

## 2. Research for Spike protein on SARS-CoV-2 virus (07/2023-08/2023)

Position: Data Scientist Intern <br>
Location: Shenzhen Bay Laboratory. <br>
tutor: Chaowang https://www.szbl.ac.cn/en/scientificresearch/researchteam/3372.html

### 2.1 Background & Explain
I was mainly involved in identifying mutational hotspots and capturing the mutation distribution using the Gaussian process on the SARS-CoV-2 spike protein, which is a key protein for global pendemics. Mutational hotspots are the places on the protein where the mutation frequency is higher than in other places, they are the main consideration for designing experiments and vaccines. Furthermore, we would like to capture the regression patterns for potential prediction analysis from biology perspective. 

Main final result is the reduced MSE of 17%, improved interpretability of the model, and cut experimental overhead by 30%  

Alongside with this project, I also productionized ETL pipelines to handle 1M+ global pandemic records sourced from 27 heterogeneous biological databases, utilizing Apache Spark for scalable job scheduling and dataflow optimization. Furthermroe used Docker to streamline deployment and support CI/CD integration for a real-time Tableau dashboard of for researchers.

As the final deployment of code is collborating with my teammates in another repository, I only demonstrate some initial coding related to Gaussian Process (mainly in the file Gaussian Process xxx.py) and hotspot indentification (the file Hotspots by kmeans method, which has not been selected for the final deployment) in the file that is fully developed by me. 

### 2.2 Algorithm design for locating **infectious hotspots**
Designed an algorithm for locating **infectious hotspots**, combining the following statistical and spatial methods:

- **Random Permutation Testing**  
- **Kernel Density Estimation (KDE)**  
- **Monte Carlo Simulations**  
- **Hierarchical Clustering Analysis (HCA)**

These methods enhanced the robustness of the spatial analysis by capturing uncertainty and spatial dependencies in mutation distributions.



### 2.3 Applying the Gaussian process for capturing the mutational distribution of the protein.
Check the 'Gaussian process for mutation distribution'. Data processing here includes Log transformation and Box-Cox transformation for 'Mutation Number'. I first implemented the basic GP method with the kernel function using Matern, a generalization of RBF, and using Random search for the parameters in the kernel function. This basic model gives a tragic result. 

I thus improved the model by following methods:<br>
#### 2.3.1. Feature Engineering:
Extracted spatial features from `(X, Y, Z)`:
- `x²`, `y²`, `z²`, `xy`, `xz`, `yz`, `sum_xyz`, `distance_to_center`

Performed feature attribution using:

- **SHAP** (Shapley Additive Explanations)  
- **Permutation Importance**

Final features selected via:

- Correlation Analysis  
- SelectKBest

**Selected features:** `x`, `y`, `x²`, `xy`, `xz`, `distance_to_center`  
**Result:** Reduced model **MSE by 17%**, decreasing lab validation efforts by **30%**

---

#### 2.3.2. Model Optimization:
Initial model with **Matern Kernel** and **Random Search** yielded poor results.

Switched to **Bayesian Optimization**, which provided:

- Smarter hyperparameter tuning  
- Surrogate modeling  
- Faster convergence  
- Improved GP kernel selection under uncertainty

---

#### 2.3.3. Data Augmentation:
Addressed **mutation imbalance** with **SMOTE (Synthetic Minority Over-sampling Technique)**:

- Generated underrepresented high-mutation samples between existing minority class points
- Verified effectiveness via **learning curve analysis**

<p align="center">
  <img src="https://github.com/bimzty/bimzty.github.io/blob/main/Photos/GP%20without%20Augmentation.png" width="400" />
  <img src="https://github.com/bimzty/bimzty.github.io/blob/main/Photos/GP%20woth%20Augmentation.png" width="400" />
</p>
<p align="center">
  <em>The residues between regression value and real value when without augmentation (left) and with augmentation</em>
</p>
  
***

## 3. Competition: Formula Student Electric, won National Third Prize (12/2020-07/2022)
Responsibility: Data Analyst and Simulation Technician<br>
The 'Track Simulation' folder consists of Python codes mainly developed to stimulate racecar performance on Track. It consists of functions for: <br>
Straight road, when the car accelerates only;<br>
Corner, when the car runs the maximum speed under the condition;<br>
Brake, when the car decelerates only;

<p align="center">
  <img src="https://github.com/bimzty/bimzty.github.io/blob/main/Photos/Track%20Simulations.png" width="300" />
</p>
<p align="center">
 <em>Car performance simulation on one track</em>
</p>

***

## 4. Coursework during Undergraduate 
### 4.1 Coursework for the module Machine Learning (12/2023)
The coursework aims to make use of the machine learning techniques learned in this course to diagnose breast cancer using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Based on the recommended model and parameters of a similar competition held by Kaggle, I mainly built 7 models and conducted a systematic approach to choosing the best one from them. 

### 4.2 Coursework for the module Introduction to Scientific Computation (09/2022-05/2023)
This course aims to introduce the concept of numerical approximation to problems that cannot be solved analytically and to develop skills in Python by implementing numerical methods. Topics included in those works are: Solving nonlinear equations (approximately) using root finding methods and analyzing their convergence; Solving linear systems of equations using direct methods and iterative techniques, including Gaussian elimination and Jacobi & Gauss-Seidel method; Approximating functions by polynomial interpolants (Lagrange polynomials), and analyzing their accuracy; Approximating derivatives and definite integrals using numerical differentiation and integration such as trapezoidal, Simpson & midpoint rule, and analyzing their convergence; Approximating ODEs using numerical methods including Euler’s method and s, higher-order RK methods.

### 4.3 Coursework for the module Statistical Models and Methods (05/2023)
The objective of the coursework is to build a predictive model for body fat content using 10 body measurement variables. we first do some exploratory analysis of the data. Secondly, we do model selection to find the best subset of variables for regression based on AIC/BIC, Mallow’s Cp, and Adjusted
R-squared criterion. Thirdly, we identify and analyze outliers and high-leverage points. Fourthly, we check the linear
model assumption by plotting the QQ-plot, residual, component residual plot, etc, and do the manipulation to a model
based on that. A comparison between our best model and the full model using test data will also be provided.

















