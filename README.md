# bimzty.github.io
This webpage stores the projects that I have done during my undergraduate

# 1.Research for Spike protein on SARS-CoV-2 virus 

Location: Shenzhen Bay Laboratory.
tutor: Chaowang https://www.szbl.ac.cn/en/scientificresearch/researchteam/3372.html

I was mainly involved in identifying mutational hotspots and capturing the mutation distribution using the Gaussian Process on the SARS-CoV-2 spike protein.

# Background & Explain

# 1.1 Applying a weighted average proximity scoring function for identifying hotspots
Check the 'Identifying hotspots use WAP method' Rmd file.

# 1.2 Applying K-means or DBSCAN method for identifying hotspots
Check the 'Identifying hotspots using Clustering method' R file for finding hotspots. Firstly, this method performs Data Preprocessing to transform the spatial position('X','Y','Z') of residuals and VirusPercentage to the same scale[add more??????]. Secondly, this method examines that the data distribution invariant before and after the process. Thirdly, the method uses K-means to cluster different residues. Fourthly, I perform the Permutation methods to examine the significance of mutation frequency for different clusters. Various hypothesis tests have been conducted in this step. We finally utilize T-SNE for dimensionality reduction and visualization.






