# Load necessary libraries
#Due the requirement of my colleague, I only write Chinese comment for this code.

library(sp)
library(spgwr)
library(spdep)
library(readxl)
library(spatialreg)
library(gstat)
library(stats)
library(dbscan)
library(cluster)
library(ggplot2)
library(scatterplot3d)
library(rgl)
library(mclust)
library(plotly)
library(dplyr)
library(FNN)
library(matrixStats)
library(tidyr)
library(scales)
library(factoextra)
library(spatstat)
library(clValid)


data<-as.data.frame(read.csv("data1.csv"))

data<-na.omit(data)
data_matrix<-as.matrix(data)
data2<-data[1:972,c("X","Y","Z","virusPercent")]
log_virus_percent <- log(data2$virusPercent)

data <- read.csv("data1.csv")
sub_data<-data[1:972,c("X","Y","Z","virusPercent")]
sub_data_1<-data[1:972,c("X","Y","Z")]
summary(sub_data)

# scatter plot
library(plot3D)
scatter3D(sub_data[, 1], sub_data[, 2], sub_data[, 3], col = "blue")

# histogram
hist(sub_data[, 1], breaks = 20)
hist(sub_data[, 2], breaks = 20)
hist(sub_data[, 3], breaks = 20)

# kernel density 
plot(density(sub_data[, 1]))
plot(density(sub_data[, 2]))
plot(density(sub_data[, 3]))
# QQ plot
qqnorm(sub_data[, 1])
qqline(sub_data[, 1])
qqnorm(sub_data[, 2])
qqline(sub_data[, 2])
qqnorm(sub_data[, 3])
qqline(sub_data[, 3])

apply(sub_data,2,mean)
apply(sub_data,2,sd)

# t test
t.test(sub_data[, 1], sub_data[, 2])
t.test(sub_data[, 1], sub_data[, 3])
t.test(sub_data[, 2], sub_data[, 3])
t.test(sub_data[, 1], sub_data[, 4])
t.test(sub_data[, 2], sub_data[, 4])
t.test(sub_data[, 3], sub_data[, 4])
# If the p-value is small, there is a significant difference between the mean and standard deviation between the two dimensions, # Further operations such as standardization are required

# Variance analysis space difference
data_df<-as.data.frame(sub_data_1)

cor_mat <- cor(data_df, method = "pearson")
print(cor_mat)
# The correlation coefficient ranges from -1 to 1, # The greater the absolute value, the stronger the correlation, while the plus or minus sign indicates the direction of the correlation
for (i in 1:(ncol(data_df) - 1)) {
  for (j in (i + 1):ncol(data_df)) {
    cor_test <- cor.test(data_df[, i], data_df[, j], method = "pearson")
    cat("correlation between", names(data_df)[i], "and", names(data_df)[j], "is", cor_test$estimate, "with p-value", cor_test$p.value, "\n")
  }
}

# Spearman
cor_mat_1 <- cor(data_df, method = "spearman")
print(cor_mat_1)

for (i in 1:(ncol(data_df) - 1)) {
  for (j in (i + 1):ncol(data_df)) {
    cor_test <- cor.test(data_df[, i], data_df[, j], method = "spearman")
    cat("correlation between", names(data_df)[i], "and", names(data_df)[j], "is", cor_test$estimate, "with p-value", cor_test$p.value, "\n")
  }
}

# Calculate Kendall grade correlation coefficient
cor_mat_2 <- cor(data_df, method = "kendall")
print(cor_mat_2)
for (i in 1:(ncol(data_df) - 1)) {
  for (j in (i + 1):ncol(data_df)) {
    cor_test <- cor.test(data_df[, i], data_df[, j], method = "kendall")
    cat("correlation between", names(data_df)[i], "and", names(data_df)[j], "is", cor_test$estimate, "with p-value", cor_test$p.value, "\n")
  }
}

#In summary:
#A small significance level indicates significant correlation between variables.
#Significant differences among data dimensions may warrant standardization.
#It is important to assess the distributional characteristics of data before and after standardization to determine its suitability for the dataset.

#特征向量+Kmean聚类法模型

#抽提坐标数据
xyz<-data2[,c("X","Y","Z")]
#标准化xyz数据
xyz_norm<-scale(xyz)
# 将标准化处理过的坐标和突变频率组合为特征向量
feature_vector <- cbind(xyz_norm, data2$virusPercent)

#抽提突变频率
w<-data2[,c("virusPercent")]
#标准化xyz数据
w_norm<-scale(w)

#library(MASS)
#library(car)

freq_trans_1 <- log(data2$virusPercent) #对数变换
freq_trans_2<-scale(freq_trans_1)
freq_trans_3<-(freq_trans_1 - min(freq_trans_1)) / (max(freq_trans_1) - min(freq_trans_1))
# 绘制变换前后的数据分布
par(mfrow=c(1,3))
#hist(data2$mutation, main="Before Transformation", xlab="freq")
hist(freq_trans_1)
hist(freq_trans_2)
hist(freq_trans_3)
# 将标准化处理过的坐标和突变频率组合为特征向量
scale_vector <- cbind(xyz_norm, freq_trans_2)
scale_vector<-data.frame(scale_vector)

#修改1 可视化标准化后的蛋白质结构
# 创建3D散点图
plot3d(feature_vector[,1], feature_vector[,2], feature_vector[,3], col = "blue", size = 2)
plot3d(data2[,1], data2[,2], data2[,3], col = "blue", size = 2)
#观察发现，原始数据是实际的972个点的三维坐标xyz和每个点的突变频率，可以画出来一个三维散点图，
#现在将xyz进行标准化处理后，得到了一个新的数据，又可以画出一个新的标准化后的三维散点图。
#前后两个散点图空间结构和形状上几乎相同。
#标准化处理会改变原始数据的数值范围和分布，但不会改变数据的结构和形状。
#因此，将原始数据进行标准化处理后，生成的新的标准化三维散点图与原始三维散点图相似，但数值范围和分布不同。

#原始的xyz和标准化后的xyz的数据范围和分布
# 查看原始数据的数据范围和分布情况
summary(data2[, 1:3])  # 显示前三列（x、y、z）的数据范围和分布情况
hist(data2[, 1])      # 绘制x列的直方图
hist(data2[, 2])      # 绘制y列的直方图
hist(data2[, 3])      # 绘制z列的直方图
hist(freq_trans_1)      #绘制virusPercent的直方图

# 查看标准化后的数据的数据范围和分布情况
summary(feature_vector[, 1:3])  # 显示前三列（x_norm、y_norm、z_norm）的数据范围和分布情况
hist(feature_vector[, 1])      # 绘制x_norm列的直方图
hist(feature_vector[, 2])      # 绘制y_norm列的直方图
hist(feature_vector[, 3])      # 绘制z_norm列的直方图
hist(scale_vector[, 4])        #绘制vrius_norm的直方图
#验证标准化前后数据分布是否相似
#可以比较前后的直方图，观察它们的形状、中心位置、分散程度等特征是否相似
#如果两个直方图的形状和特征较为相似，那么它们的分布也可能相似
#另外，还可以使用一些统计方法来比较两个数据集的分布情况，例如Kolmogorov-Smirnov检验、Anderson-Darling检验，
#这些方法可以计算两个数据集之间的距离或差异程度，并给出相应的显著性水平，可以用来评估前后数据分布的相似性。

# 加载ks包，用于进行Kolmogorov-Smirnov检验
library(ks)

# 对比前后x列数据的分布
ks.test(data2[, "X"], feature_vector[, "X"])  # 进行Kolmogorov-Smirnov检验，比较x列的分布

# 对比前后y列数据的分布
ks.test(data2[, 2], feature_vector[, 2])  # 进行Kolmogorov-Smirnov检验，比较y列的分布

# 对比前后z列数据的分布
ks.test(data2[, 3], feature_vector[, 3])  # 进行Kolmogorov-Smirnov检验，比较z列的分布

#如果两个数据列的分布相似，则Kolmogorov-Smirnov检验的p值应该比较大（大于0.05），
#否则p值较小（小于0.05）。如果p值较小，则可以认为前后数据列的分布不相似

#如果标准化后的数据分布与原始数据分布不相似，
#那么将标准化后的数据进行聚类可能会导致结果不准确。
#因为聚类算法通常基于数据的距离或相似性进行聚类，如果数据分布不同，
#那么聚类结果可能会受到影响，导致聚类效果不佳。
#也就是说，如果标准化后数据的分布和标准化前的不相似的话，
#如果对标准化后的数据进行聚类，得到突变概率高发区所具有的点的话，
#不能直接说这些点在未标准化的数据中也对应突变高发区。

#另外，即使聚类结果在标准化后的数据上表现良好，也不一定能直接反映到原始数据上。
#因为标准化是一种线性变换，它可能会改变数据的分布、范围和形状等特征。
#如果将聚类结果直接应用到原始数据上，可能会导致聚类结果不准确。

#xyz三个值的数据类型是地理位置，而突变频率的数据类型是概率，
#将地理坐标和突变概率两种数据类型合在一起进行K均值聚类是不合适的，
#因为地理坐标和突变概率是不同类型的变量，它们具有不同的量纲和意义，
#合并在一起可能会导致聚类结果失真。
#为了解决这个问题，需要将地理坐标和突变概率转换为相同的尺度，
#例如将地理坐标转换为数值型变量，将突变概率转换为概率的对数或负数。

#如果数据已经是数值型的三维坐标，那么不需要再进行特殊的转换，可以直接使用这些数值型变量进行聚类分析，
#概率的对数或负数可以用来表示概率的大小，同时也可以用来比较不同概率之间的大小关系。
#例如，将突变概率p转换为概率的对数ln(p)或负数-np，可以将概率的范围从[0,1]转换为[-∞,0]或[0,∞]，
#使其与地理坐标在同一尺度下进行比较。
#这种转换方法可以将概率转换为数值型变量，使其与其他数值型变量一起参与聚类分析。

#坐标值类的数据可以被认为是数值型变量，是因为它们是由数值表达的。
#在空间坐标系中，每个点的位置都可以用一组数值来表示。
#经过标准化处理后的坐标值可以仍然被认为是数值型变量。
#标准化处理是一种线性变换，将原始数据映射到一个新的坐标系中，
#使得每个维度的数据具有相同的尺度和范围。
#在标准化处理后，坐标值的数值类型和数值特征都没有发生变化

#从密度图中观察密度曲线的形状和位置来判断标准化前后数据的分布差异性
library(ggplot2)
library(gridExtra)
# 原始数据的密度图
p1 <- ggplot(data2, aes(x = X)) +
  geom_density(fill = "steelblue", color = "white") +
  xlab("X") + ylab("Density") + ggtitle("Original X Density Distribution")

p2 <- ggplot(data2, aes(x = Y)) +
  geom_density(fill = "steelblue", color = "white") +
  xlab("Y") + ylab("Density") + ggtitle("Original Y Density Distribution")

p3 <- ggplot(data2, aes(x = Z)) +
  geom_density(fill = "steelblue", color = "white") +
  xlab("Z") + ylab("Density") + ggtitle("Original Z Density Distribution")

p7 <- ggplot(data2, aes(x = virusPercent)) +
  geom_density(fill = "steelblue", color = "white") +
  xlab("virusPercent") + ylab("Density") + ggtitle("Original vP Density Distribution")
p7<-p7+xlim(0,0.02)

feature_vector<-data.frame(feature_vector)
# 标准化后的密度图
p4 <- ggplot(feature_vector, aes(x = X)) +
  geom_density(fill = "steelblue", color = "white") +
  xlab("X") + ylab("Density") + ggtitle("Standardized X Density Distribution")

p5 <- ggplot(feature_vector, aes(x = Y)) +
  geom_density(fill = "steelblue", color = "white") +
  xlab("Y") + ylab("Density") + ggtitle("Standardized Y Density Distribution")

p6 <- ggplot(feature_vector, aes(x = Z)) +
  geom_density(fill = "steelblue", color = "white") +
  xlab("Z") + ylab("Density") + ggtitle("Standardized Z Density Distribution")

p8 <- ggplot(scale_vector, aes(x = V4)) +
  geom_density(fill = "steelblue", color = "white") +
  xlab("virusPercent") + ylab("Density") + ggtitle("standardized vP Density Distribution")
#p8<-p8+xlim(-0.2,-0.1)

# 将六张图合并为两行三列,形状，位置，峰度和偏度
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8 ,nrow = 2, ncol = 4)

#如果在不同的变量上进行标准化，即使在每个变量上的密度图相似，
#也不能保证在所有变量上的分布相似。
#因此，如果使用标准化数据进行聚类分析，建议同时考虑多个变量的分布情况，而不仅仅是单个变量。
#如果在每个变量上的密度图都相似，但KS检验却拒绝原假设，可能是由于KS检验对样本量敏感，
#并且在样本量较大时可能会拒绝原假设。此外，KS检验可能会检测到一些细微的差异，这些差异可能在实际分析中并不重要。
#因此，还可以使用其他的检验方法，例如Anderson-Darling检验或Chi-squared检验等，以进一步探究数据的分布是否相似

# 循环绘制所有的二维多变量密度图
# 标准化前的二维密度图
library(ggplot2)
p1 <- ggplot(data2, aes(X, Y)) + 
  geom_density_2d(alpha = 0.6) +
  labs(x = "X", y = "Y") +
  theme_bw()

p2 <- ggplot(data2, aes(X, Z)) + 
  geom_density_2d(alpha = 0.6) +
  labs(x = "X", y = "Y") +
  theme_bw()

p3 <- ggplot(data2, aes(Y, Z)) + 
  geom_density_2d(alpha = 0.6) +
  labs(x = "Y", y = "Z") +
  theme_bw()
#标准化后的二维变量密度图
p4 <- ggplot(feature_vector, aes(X, Y)) + 
  geom_density_2d(alpha = 0.6) +
  labs(x = "X", y = "Y") +
  theme_bw()

p5 <- ggplot(feature_vector, aes(X, Z)) + 
  geom_density_2d(alpha = 0.6) +
  labs(x = "X", y = "Z") +
  theme_bw()

p6 <- ggplot(feature_vector, aes(Y, Z)) + 
  geom_density_2d(alpha = 0.6) +
  labs(x = "Y", y = "Z") +
  theme_bw()

# 将六张图合并为两行三列,形状，位置，峰度和偏度
grid.arrange(p1, p2, p3, p4, p5, p6, nrow = 2, ncol = 3)

#将坐标和频率转化到同一尺度下
#将不同尺度的数据转化到同一尺度可以帮助我们消除不同类型的数据之间的差异，
#包括地理意义上的坐标和概率意义上的突变频率。
#这可以使得不同类型的数据在聚类分析中具有可比性和可解释性，从而提高聚类分析的准确性和有效性

#坐标值属于连续型数值尺度，通常表示物理空间中的位置或大小。
#在聚类分析中，坐标值通常被视为连续型数值特征，需要进行特征缩放或标准化处理，以便进行聚类分析。
#突变概率特征属于概率型数值尺度，通常表示某个事件发生的概率或可能性。
#在聚类分析中，突变概率特征通常被视为概率型数值特征，需要进行特征缩放或标准化处理，以便进行聚类分析

#特征缩放或标准化处理后，坐标值和突变概率特征都将被转化到相同的数值范围内，
#通常是[0,1]或[-1,1]的范围内。
#这样，它们就可以在聚类算法中进行比较和处理，而不会因为不同的度量单位和量纲而对聚类结果造成影响。

# 检查数据集中是否有NA和Infinity值
sum(is.na(data2$virusPercent))
sum(is.infinite(data2$virusPercent))
sum(is.na(scale_vector$V4))
sum(is.infinite(scale_vector$V4))

#转化方法1：同时标准化处理

# 对比前后突变频率的分布
ks.test(freq_trans_1, scale_vector[, 4])  # 进行Kolmogorov-Smirnov检验，比较z列的分布
duplicated(data2[,4])
table(data2[,4])

#标准化处理可以将不同特征的数据转化到相同的尺度和分布范围内，
#消除不同特征之间的单位和量纲差异。
#因此，如果同时对坐标值和突变频率进行标准化处理，处理前后数据分布相似，
#那么可以认为已经将它们转化到了相同的尺度下。
#标准化处理将数据转化为均值为0，标准差为1的分布范围内，
#这意味着数据的取值范围已经被缩放到相同的尺度。
#如果处理前后数据分布相似，说明它们的方差和分布范围已经被调整到相同的尺度，因此可以认为它们已经被转化到了相同的尺度下。

#转化方法2：特征放缩
# 最小-最大缩放x，y，z坐标值
data2$x <- (data2$X - min(data2$X)) / (max(data2$X) - min(data2$X))
data2$y <- (data2$Y - min(data2$Y)) / (max(data2$Y) - min(data2$Y))
data2$z <- (data2$Z - min(data2$Z)) / (max(data2$Z) - min(data2$Z))
data2$mutation <- (data2$virusPercent - min(data2$virusPercent)) / (max(data2$virusPercent) - min(data2$virusPercent))

# 构建新的数据框只存放缩后的四列数据
new_data <- data.frame(x = data2$x, y = data2$y, z = data2$z, mutation = freq_trans_3)
#验证特征放缩前后数据分布相似性

# 创建3D散点图
plot3d(new_data[,1], new_data[,2], new_data[,3], col = "blue", size = 2)

# 绘制散点图矩阵
pairs(data2[, c("X", "Y", "Z", "virusPercent")], main = "Scatterplot Matrix before")
pairs(new_data[,c("x","y","z","mutation")],main = "Scatterplot Matrix after")

# 比较缩放前后数据的直方图
par(mfrow = c(2, 4))   # 将图形区域划分为2行4列
hist(log(data2$virusPercent), main = "Mut Before Scale", xlab = "Mutation")
hist(data2$X, main = "X Before Scale", xlab = "X")
hist(data2$Y, main = "Y Before Scale", xlab = "Y")
hist(data2$Z, main = "Z Before Scale", xlab = "Z")
hist(new_data$mutation, main = "Mut After Scale", xlab = "Mutation")
hist(new_data$x, main = "X After Scale", xlab = "X")
hist(new_data$y, main = "Y After Scale", xlab = "Y")
hist(new_data$z, main = "Z After Scale", xlab = "Z")

# 比较缩放前后数据的密度图
par(mfrow = c(2, 4))   # 将图形区域划分为2行4列
plot(density(log(data2$virusPercent)), main = "Mut Before Scale", xlab = "Mutation")
plot(density(data2$X), main = "X Before Scaling", xlab = "X")
plot(density(data2$Y), main = "Y Before Scaling", xlab = "Y")
plot(density(data2$Z), main = "Z Before Scaling", xlab = "Z")
plot(density(new_data$mutation), main = "Mut After Scale", xlab = "Mutation")
plot(density(new_data$x), main = "X After Scale", xlab = "X")
plot(density(new_data$y), main = "Y After Scale", xlab = "Y")
plot(density(new_data$z), main = "Z After Scale", xlab = "Z")

# 比较缩放前后数据的QQ图
par(mfrow = c(2, 4))   # 将图形区域划分为2行4列
qqnorm(log(data2$virusPercent), main = "Mut Before Scale")
qqline(log(data2$virusPercent))
qqnorm(data2$X, main = "X Before Scale")
qqline(data2$X)
qqnorm(data2$Y, main = "Y Before Scale")
qqline(data2$Y)
qqnorm(data2$Z, main = "Z Before Scale")
qqline(data2$Z)
qqnorm(new_data$mutation, main = "Mut After Scale")
qqline(new_data$mutation)
qqnorm(new_data$x, main = "X After Scale")
qqline(new_data$x)
qqnorm(new_data$y, main = "Y After Scale")
qqline(new_data$y)
qqnorm(new_data$z, main = "Z After Scale")
qqline(new_data$z)

#如果我现在如果发现标准化前后数据分布几乎相似的话，
#不能直接说明可以将数据标准化后去聚类得到的突变频率高的点可以直接对应到标准化前的点上。
#如果想要将标准化后的聚类结果对应到原始数据上，
#可以使用聚类算法的聚类中心或簇标记等信息，结合标准化前后的数据分布情况，来确定标准化后每个聚类簇所对应的原始数据点


# 进行 KMeans 聚类
set.seed(123)
kmeans_result <- kmeans(new_data, centers = 3, nstart = 20)
kmeans_result_1 <- kmeans(scale_vector, centers = 3, nstart = 20)
# 将聚类结果添加到数据框中
data2$cluster <- as.factor(kmeans_result$cluster)
# 可视化聚类结果&对聚类结果进一步分析
fviz_cluster(list(data = new_data, cluster = kmeans_result$cluster))
fviz_cluster(list(data = scale_vector, cluster = kmeans_result_1$cluster))

library(dplyr)

# 统计每个聚类簇中数据点的数量
new_data$cluster <- as.factor(kmeans_result$cluster)
cluster_count <- new_data %>%
  group_by(cluster) %>%
  summarize(count=n())

# 计算每个聚类簇中数据点的比例
cluster_count$proportion <- cluster_count$count / nrow(new_data)
print(cluster_count)

library(ggplot2)

# 绘制聚类簇的频率分布直方图
ggplot(cluster_count, aes(x=cluster, y=count)) +
  geom_bar(stat="identity", fill="blue") +
  labs(x="Cluster", y="Count", title="Cluster Frequency Distribution")

# 绘制聚类簇的密度图
ggplot(new_data, aes(x=x, y=y, fill=cluster)) +
  geom_density_2d() +
  scale_fill_manual(values=c("red", "blue", "green", "yellow")) +
  labs(x="X", y="Y", fill="Cluster", title="Cluster Density Plot")

# 比较聚类结果和突变频率
cluster_freq <- data2 %>%
  group_by(cluster) %>%
  summarise(mean_freq = mean(virusPercent), count = n())

total_freq <- mean(data2$virusPercent)

print(cluster_freq)
print(total_freq)


#Permutation法(排名百分数假设检验+特征放缩处理)
# 计算原始数据的聚类结果
orig_clusters <- kmeans(new_data, 122)
n_cluster<-length(unique(orig_clusters$cluster)) 
data3<-data[1:972,c("X","Y","Z","virusPercent")]
orig_means <- tapply(new_data$mutation, orig_clusters$cluster, mean)
orig_pcts <- round(rank(orig_means)/length(orig_means)*100, 2)

# 进行2000次突变概率重排并计算每次聚类结果
n_permutations <- 2000
perm_results <- matrix(NA, n_permutations, length(orig_means))
for (i in 1:n_permutations) {
  perm_data <- data3
  perm_data[,ncol(data3)] <- sample(data3[,ncol(data3)], replace=FALSE)
  perm_data[,ncol(data3)] <- log(perm_data[,ncol(data3)])
  mut_scale <- (perm_data[,ncol(data3)] - min(perm_data[,ncol(data3)])) / (max(perm_data[,ncol(data3)]) - min(perm_data[,ncol(data3)]))
  sta_scale <- data.frame(x = data2$x,y = data2$y,z = data2$z,mutation = mut_scale)
  perm_clusters <- kmeans(sta_scale, 122)
  perm_means <- tapply(sta_scale$mutation, perm_clusters$cluster, mean)
  perm_pcts <- round(rank(perm_means)/length(perm_means)*100, 2)
  perm_results[i,] <- perm_pcts
}

# 计算每个聚类的p值
p_values <- numeric(length(orig_means))
for (i in 1:length(orig_means)) {
  p_values[i] <- mean(perm_results[,i] >= orig_pcts[i])
}

# 输出结果
cat("原始聚类结果的均值：", orig_means, "\n")
cat("原始聚类结果的百分位数：", orig_pcts, "\n")
cat("p值：", p_values, "\n")
cat("原始聚类结果中最高的平均突变频率：", max(orig_means), "\n")
cat("对应的p值：", p_values[which.max(orig_means)], "\n")

# 如果最高平均突变频率的类的p值小于0.05，则显示该类的点
if (p_values[which.max(orig_means)] < 0.05) {
  cat("原始聚类结果中平均突变频率最高的聚类中的点：\n")
  cat(paste("行号\tX\tY\tZ\tvirusPercent\n"))
  selected_rows <- which(orig_clusters$cluster == which.max(orig_means))
  for (i in selected_rows) {
    cat(paste(i, "\t", new_data[i, "x"], "\t", new_data[i, "y"], "\t", new_data[i, "z"], "\t", new_data[i, "mutation"], "\n"))
  }
  
  significant_points <- which(orig_clusters$cluster == which.max(orig_means))  # 提取p值小于0.05的聚类中的所有数据点的索引
  print(significant_points)
  # 提取p值小于0.05的聚类中的所有数据点的坐标 
  significant_coordinates <- new_data[significant_points, c("x", "y", "z")] 
  
  # 在三维点图中可视化所有数据点和p值小于0.05的聚类中的数据点 
  fig <- plot_ly() %>% 
    add_trace(data = new_data, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = "All Points") %>% 
    layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z"))) 
  
  fig 
}

# 循环遍历每个聚类
i_num <- list()
j <- 1
class_mean_distances <- numeric(n_cluster)
class_mean_mutation_rates <- numeric(n_cluster)
colors <- heat.colors(n_cluster)
for (i in 1:n_cluster){
  # 如果p-value小于0.05，则将当前聚类的索引添加到significant_clusters向量中
  if (p_values[i] < 0.05) {
    cat("p值：", p_values[i], "\n")
    
    # 提取属于当前聚类的数据点的索引
    cluster_points <- which(orig_clusters$cluster == i)
    cat("类数：", i, "\n")
    cat("原子序数：", data$atomSerialNumber[cluster_points], "\n")
    cat("该类均值：", orig_means[i],"\n")
    # 提取属于当前聚类的数据点的坐标
    cluster_coordinates <- new_data[cluster_points, c("x", "y", "z")]
    # 为聚类指定颜色
    color <- colors[i]
    i_num[j]<-i
    j = j+1
    class_points <- new_data[orig_clusters$cluster == i,]
    # 计算该类中所有点之间的欧几里得距离
    class_distances <- as.matrix(dist(class_points[,1:3]))
    class_distances_vector <- as.vector(class_distances)
    class_distances_vector <- class_distances_vector[class_distances_vector != 0]
    
    # 输出该类中所有点之间距离的统计量
    cat("Class", i, "distance statistics:\n")
    cat("Mean distance:", mean(class_distances_vector), "\n")
    cat("Median distance:", median(class_distances_vector), "\n")
    cat("Minimum distance:", min(class_distances_vector), "\n")
    cat("Maximum distance:", max(class_distances_vector), "\n")
    class_mean_distances[i] <- mean(class_distances_vector)
    
    # 计算平均突变率
    class_mean_mutation_rates[i] <- mean(new_data[orig_clusters$cluster == i, "mutation"])
    # 在三维散点图中添加聚类
    fig <- fig %>% add_trace(data = cluster_coordinates, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = paste("Cluster ", i), marker = list(color = color))
  }
}
# 设定散点图的布局
fig <- fig %>% layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z")))
# 显示散点图
fig

# 将平均距离和平均突变率存储在一个数据框中
class_summary <- data.frame(mean_distance = class_mean_distances, mean_mutation_rate = class_mean_mutation_rates)
# 使用ggplot2包绘制折线图
library(ggplot2)
ggplot(class_summary, aes(x = mean_distance, y = mean_mutation_rate)) + geom_line() + geom_point() + xlab("Mean Distance") + ylab("Mean Mutation Rate") + ggtitle("Mean Distance vs. Mean Mutation Rate")
#讲所有类标签存储到表格中
i_num <- data.frame(i_num)
i_num <- as.numeric(i_num)
# 提取所有显著性聚类的点的索引
significant_pointer <- which(orig_clusters$cluster %in% i_num)
# 提取所有显著性聚类的点的xyz和突变频率
significant_data <- new_data[significant_pointer, c("x", "y", "z", "mutation")]
significant_num <- data[significant_pointer, c("virusNumber")]
# 输出结果
print(significant_data)
print(significant_num)
# 进行层次聚类分析
mut_region_dist <- dist(significant_data, method = "euclidean")
mut_region_hclust <- hclust(mut_region_dist, method = "ward.D2")

# 将聚类结果可视化为聚类热图
library(pheatmap)
pheatmap(significant_data, 
         cluster_rows = mut_region_hclust, 
         cluster_cols = FALSE,
         color = colorRampPalette(c("white", "blue"))(50),
         main = "High Mutation Regions Clustering Heatmap",
         fontsize = 8)









#(特征放缩+label重排)
#Permutation法(排名百分数假设检验+特征放缩处理)
# 计算原始数据的聚类结果
k <- 122
perm_means <- numeric(k)
orig_clusters <- kmeans(new_data, k)
n_cluster<-length(unique(orig_clusters$cluster)) 
new_data$cluster <- as.factor(orig_clusters$cluster)
orig_means <- tapply(new_data$mutation, orig_clusters$cluster, mean)
orig_pcts <- round(rank(orig_means)/length(orig_means)*100, 2)

# 进行2000次突变概率重排并计算每次聚类结果
n_permutations <- 2000
perm_results <- matrix(NA, n_permutations, length(orig_means))
for (i in 1:n_permutations) {
  perm_data <- new_data
  perm_data[,ncol(new_data)] <- sample(new_data[,ncol(new_data)], replace=FALSE)
  sta_scale <- data.frame(x = data2$x,y = data2$y,z = data2$z,mutation = freq_trans_3,cluster = perm_data[,ncol(new_data)])
  perm_means <- aggregate(sta_scale[, "mutation"], by = list(sta_scale$cluster), mean)$x
  perm_pcts <- round(rank(perm_means)/length(perm_means)*100, 2)
  perm_results[i,] <- perm_pcts
}

# 计算每个聚类的p值
p_values <- numeric(length(orig_means))
for (i in 1:length(orig_means)) {
  p_values[i] <- mean(perm_results[,i] >= orig_pcts[i])
}

# 输出结果
cat("原始聚类结果的均值：", orig_means, "\n")
cat("原始聚类结果的百分位数：", orig_pcts, "\n")
cat("p值：", p_values, "\n")
cat("原始聚类结果中最高的平均突变频率：", max(orig_means), "\n")
cat("对应的p值：", p_values[which.max(orig_means)], "\n")

# 如果最高平均突变频率的类的p值小于0.05，则显示该类的点
if (p_values[which.max(orig_means)] < 0.05) {
  cat("原始聚类结果中平均突变频率最高的聚类中的点：\n")
  cat(paste("行号\tX\tY\tZ\tvirusPercent\n"))
  selected_rows <- which(orig_clusters$cluster == which.max(orig_means))
  for (i in selected_rows) {
    cat(paste(i, "\t", new_data[i, "x"], "\t", new_data[i, "y"], "\t", new_data[i, "z"], "\t", new_data[i, "mutation"], "\n"))
  }
  
  significant_points <- which(orig_clusters$cluster == which.max(orig_means))  # 提取p值小于0.05的聚类中的所有数据点的索引
  print(significant_points)
}
# 提取p值小于0.05的聚类中的所有数据点的坐标 
significant_coordinates <- new_data[significant_points, c("x", "y", "z")] 

# 在三维点图中可视化所有数据点和p值小于0.05的聚类中的数据点 
fig_2 <- plot_ly() %>% 
  add_trace(data = new_data, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = "All Points") %>% 
  layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z"))) 

fig_2
# 循环遍历每个聚类
i_num <- list()
j <- 1
colors <- heat.colors(n_cluster)
for (i in 1:n_cluster){
  # 如果p-value小于0.05，则将当前聚类的索引添加到significant_clusters向量中
  if (p_values[i] < 0.05) {
    cat("p值：", p_values[i], "\n")
    
    # 提取属于当前聚类的数据点的索引
    cluster_points <- which(orig_clusters$cluster == i)
    cat("类数：", i, "\n")
    cat("原子序数：", data$atomSerialNumber[cluster_points], "\n")
    cat("该类均值：", orig_means[i],"\n")
    # 提取属于当前聚类的数据点的坐标
    cluster_coordinates <- new_data[cluster_points, c("x", "y", "z")]
    # 为聚类指定颜色
    color <- colors[i]
    i_num[j]<-i
    j = j+1
    class_points <- new_data[orig_clusters$cluster == i, ]
    # 计算该类中所有点之间的欧几里得距离
    class_distances <- as.matrix(dist(class_points[,1:3]))
    class_distances_vector <- as.vector(class_distances)
    class_distances_vector <- class_distances_vector[class_distances_vector != 0]
    
    # 输出该类中所有点之间距离的统计量
    cat("Class", i, "distance statistics:\n")
    cat("Mean distance:", mean(class_distances_vector), "\n")
    cat("Median distance:", median(class_distances_vector), "\n")
    cat("Minimum distance:", min(class_distances_vector), "\n")
    cat("Maximum distance:", max(class_distances_vector), "\n")
    class_mean_distances[i] <- mean(class_distances_vector)
    
    # 计算平均突变率
    class_mean_mutation_rates[i] <- mean(new_data[orig_clusters$cluster == i, "mutation"])
    # 在三维散点图中添加聚类
    fig_2 <- fig_2 %>% add_trace(data = cluster_coordinates, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = paste("Cluster ", i), marker = list(color = color))
  }
}
# 设定散点图的布局
fig_2 <- fig_2 %>% layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z")))
# 显示散点图
fig_2

# 将平均距离和平均突变率存储在一个数据框中
class_summary <- data.frame(mean_distance = class_mean_distances, mean_mutation_rate = class_mean_mutation_rates)
# 使用ggplot2包绘制折线图
ggplot(class_summary, aes(x = mean_distance, y = mean_mutation_rate)) + geom_line() + geom_point() + xlab("Mean Distance") + ylab("Mean Mutation Rate") + ggtitle("Mean Distance vs. Mean Mutation Rate")

#讲所有类标签存储到表格中
i_num <- data.frame(i_num)
i_num <- as.numeric(i_num)
# 提取所有显著性聚类的点的索引
significant_pointer <- which(orig_clusters$cluster %in% i_num)
# 提取所有显著性聚类的点的xyz和突变频率
significant_data <- new_data[significant_pointer, c("x", "y", "z", "mutation")]
# 输出结果
print(significant_data)






#Permutation法(排名百分数假设检验+标准化处理)
# 计算原始数据的聚类结果
orig_clusters <- kmeans(scale_vector, 122)
n_cluster<-length(unique(orig_clusters$cluster))
data3<-data[1:972,c("X","Y","Z","virusPercent")]
orig_means <- tapply(scale_vector$V4, orig_clusters$cluster, mean)
orig_pcts <- round(rank(orig_means)/length(orig_means)*100, 2)

# 进行2000次突变概率重排并计算每次聚类结果
n_permutations <- 2000
perm_results <- matrix(NA, n_permutations, length(orig_means))
x_scale <- scale(data3[,1])
y_scale <- scale(data3[,2])
z_scale <- scale(data3[,3])
for (i in 1:n_permutations) {
  perm_data <- data3
  perm_data[,ncol(data3)] <- sample(data3[,ncol(data3)], replace=FALSE)
  perm_data[,ncol(data3)] <- log(perm_data[,ncol(data3)])
  mut_scale <- scale(perm_data[,ncol(data3)])
  sta_scale <- data.frame(x = x_scale,y = y_scale,z = z_scale,mutation = mut_scale)
  perm_clusters <- kmeans(sta_scale, 122)
  perm_means <- tapply(sta_scale$mutation, perm_clusters$cluster, mean)
  perm_pcts <- round(rank(perm_means)/length(perm_means)*100, 2)
  perm_results[i,] <- perm_pcts
}
# 计算每个聚类的p值
p_values <- numeric(length(orig_means))
for (i in 1:length(orig_means)) {
  p_values[i] <- mean(perm_results[,i] >= orig_pcts[i])
}
# 输出结果
cat("原始聚类结果的均值：", orig_means, "\n")
cat("原始聚类结果的百分位数：", orig_pcts, "\n")
cat("p值：", p_values, "\n")
cat("原始聚类结果中最高的平均突变频率：", max(orig_means), "\n")
cat("对应的p值：", p_values[which.max(orig_means)], "\n")

# 如果最高平均突变频率的类的p值小于0.05，则显示该类的点
if (p_values[which.max(orig_means)] < 0.05) {
  cat("原始聚类结果中平均突变频率最高的聚类中的点：\n")
  cat(paste("行号\tX\tY\tZ\tvirusPercent\n"))
  selected_rows <- which(orig_clusters$cluster == which.max(orig_means))
  for (i in selected_rows) {
    cat(paste(i, "\t", scale_vector[i, "X"], "\t", scale_vector[i, "Y"], "\t", scale_vector[i, "Z"], "\t", scale_vector[i, "V4"], "\n"))
  }
  
  # 可视化所有点（用蓝色表示）和突出显示的点（用红色表示）
  significant_points <- which(orig_clusters$cluster == which.max(orig_means))  # 提取p值小于0.05的聚类中的所有数据点的索引
  print(significant_points)
  # 提取p值小于0.05的聚类中的所有数据点的坐标 
  significant_coordinates <- scale_vector[significant_points, c("X", "Y", "Z")] 
  
  # 在三维点图中可视化所有数据点和p值小于0.05的聚类中的数据点 
  fig_1 <- plot_ly() %>% 
    add_trace(data = scale_vector, x = ~X, y = ~Y, z = ~Z, type = "scatter3d", mode = "markers", name = "All Points") %>% 
    layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z"))) 
  
  fig_1
}
# 循环遍历每个聚类
colors <- heat.colors(n_cluster)
for (i in 1:n_cluster){
  # 如果p-value小于0.05，则将当前聚类的索引添加到significant_clusters向量中
  if (p_values[i] < 0.05) {
    cat("p值：", p_values[i], "\n")
    
    # 提取属于当前聚类的数据点的索引
    cluster_points <- which(orig_clusters$cluster == i)
    cat("类数：", i, "\n")
    cat("原子序数：", cluster_points, "\n")
    cat("该类均值：", orig_means[i],"\n")
    # 提取属于当前聚类的数据点的坐标
    cluster_coordinates <- scale_vector[cluster_points, c("X", "Y", "Z")]
    # 为聚类指定颜色
    color <- colors[i]
    # 在三维散点图中添加聚类
    fig_1 <- fig_1 %>% add_trace(data = cluster_coordinates, x = ~X, y = ~Y, z = ~Z, type = "scatter3d", mode = "markers", name = paste("Cluster ", i), marker = list(color = color))
  }
}
# 设定散点图的布局
fig_1 <- fig_1 %>% layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z")))
# 显示散点图
fig_1


#(标准化+label重排)
#Permutation法(排名百分数假设检验+特征放缩处理)
# 计算原始数据的聚类结果
k <- 122
perm_means <- numeric(k)
orig_clusters <- kmeans(scale_vector, k)
n_cluster<-length(unique(orig_clusters$cluster)) 
scale_vector$cluster <- as.factor(orig_clusters$cluster)
orig_means <- tapply(scale_vector$V4, orig_clusters$cluster, mean)
orig_pcts <- round(rank(orig_means)/length(orig_means)*100, 2)

# 进行2000次突变概率重排并计算每次聚类结果
n_permutations <- 2000
perm_results <- matrix(NA, n_permutations, length(orig_means))
for (i in 1:n_permutations) {
  perm_data <- scale_vector
  perm_data[,ncol(scale_vector)] <- sample(scale_vector[,ncol(scale_vector)], replace=FALSE)
  sta_scale <- data.frame(X = x_scale, Y = y_scale, Z = z_scale, V4 = freq_trans_2,cluster = perm_data[,ncol(scale_vector)])
  perm_means <- aggregate(sta_scale[, "V4"], by = list(sta_scale$cluster), mean)$x
  perm_pcts <- round(rank(perm_means)/length(perm_means)*100, 2)
  perm_results[i,] <- perm_pcts
}

# 计算每个聚类的p值
p_values <- numeric(length(orig_means))
for (i in 1:length(orig_means)) {
  p_values[i] <- mean(perm_results[,i] >= orig_pcts[i])
}

# 输出结果
cat("原始聚类结果的均值：", orig_means, "\n")
cat("原始聚类结果的百分位数：", orig_pcts, "\n")
cat("p值：", p_values, "\n")
cat("原始聚类结果中最高的平均突变频率：", max(orig_means), "\n")
cat("对应的p值：", p_values[which.max(orig_means)], "\n")

# 如果最高平均突变频率的类的p值小于0.05，则显示该类的点
if (p_values[which.max(orig_means)] < 0.05) {
  cat("原始聚类结果中平均突变频率最高的聚类中的点：\n")
  cat(paste("行号\tX\tY\tZ\tvirusPercent\n"))
  selected_rows <- which(orig_clusters$cluster == which.max(orig_means))
  for (i in selected_rows) {
    cat(paste(i, "\t", scale_vector[i, "X"], "\t", scale_vector[i, "Y"], "\t", scale_vector[i, "Z"], "\t", scale_vector[i, "V4"], "\n"))
  }
  
  significant_points <- which(orig_clusters$cluster == which.max(orig_means))  # 提取p值小于0.05的聚类中的所有数据点的索引
  print(significant_points)
}else{
  # 在三维点图中可视化所有数据点和p值小于0.05的聚类中的数据点 
  fig_3 <- plot_ly() %>% 
    add_trace(data = scale_vector, x = ~X, y = ~Y, z = ~Z, type = "scatter3d", mode = "markers", name = "All Points") %>% 
    layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z"))) 
  
  fig_3
}

# 循环遍历每个聚类
i_num <- list()
j <- 1
colors <- heat.colors(n_cluster)
for (i in 1:n_cluster){
  # 如果p-value小于0.05，则将当前聚类的索引添加到significant_clusters向量中
  if (p_values[i] < 0.05) {
    cat("p值：", p_values[i], "\n")
    
    # 提取属于当前聚类的数据点的索引
    cluster_points <- which(orig_clusters$cluster == i)
    cat("类数：", i, "\n")
    cat("原子序数：", data$atomSerialNumber[cluster_points], "\n")
    cat("该类均值：", orig_means[i],"\n")
    # 提取属于当前聚类的数据点的坐标
    cluster_coordinates <- scale_vector[cluster_points, c("X", "Y", "Z")]
    # 为聚类指定颜色
    color <- colors[i]
    i_num[j]<-i
    j = j+1
    # 在三维散点图中添加聚类
    fig_3 <- fig_3 %>% add_trace(data = cluster_coordinates, x = ~X, y = ~Y, z = ~Z, type = "scatter3d", mode = "markers", name = paste("Cluster ", i), marker = list(color = color))
  }
}
# 设定散点图的布局
fig_3 <- fig_3 %>% layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z")))
# 显示散点图
fig_3
#讲所有类标签存储到表格中
i_num <- data.frame(i_num)
i_num <- as.numeric(i_num)
# 提取所有显著性聚类的点的索引
significant_pointer <- which(orig_clusters$cluster %in% i_num)
# 提取所有显著性聚类的点的xyz和突变频率
significant_data <- scale_vector[significant_pointer, c("X", "Y", "Z", "V4")]

# 输出结果
print(significant_data)











new_scale<-data.frame(x=data2$x,y=data2$y,z = data2$z,mutation = freq_trans_3)
# 使用Rtsne进行t-SNE降维和可视化
tsne_result <- Rtsne(new_scale, dims = 2, perplexity = 30, verbose = TRUE)
plot(tsne_result$Y, pch = 20)

# 创建一个困惑度向量，包含不同的困惑度取值
perplexities <- seq(5, 50, by = 5)

# 使用多个困惑度取值进行t-SNE降维
tsne_results <- list()
for (i in seq_along(perplexities)) {
  cat("Running t-SNE with perplexity =", perplexities[i], "\n")
  tsne_results[[i]] <- Rtsne(new_scale, dims = 2, perplexity = perplexities[i], verbose = TRUE)
  cat("Time is : ", tsne_results[[i]]$times$Y_total, "\n")
}


# 绘制不同困惑度取值下的t-SNE结果和可视化结果
par(mfrow = c(2, 3))
for (i in seq_along(perplexities)) {
  plot(tsne_results[[i]]$Y, pch = 20, main = paste("Perplexity =", perplexities[i]))
}

# 运行t-SNE算法10次
tsne_results_1 <- list()
for (i in 1:10) {
  tsne_results_1[[i]] <- Rtsne(new_scale, dims = 2, perplexity = 30, verbose = TRUE)
}

# 综合结果
Y <- tsne_results_1[[1]]$Y
for (i in 2:10) {
  Y <- Y + tsne_results_1[[i]]$Y
}
Y <- Y / 10
plot(Y, pch = 20)

# 运行DBSCAN聚类算法
dbscan_cluster <- dbscan(Y, eps = 0.5, minPts = 5)

# 将聚类结果和原始数据合并为一个数据框
df <- data.frame(Y, cluster = dbscan_cluster$cluster)

# 使用ggplot2绘制散点图，并将聚类结果用颜色标识
ggplot(df, aes(x = X1, y = X2, color = factor(cluster))) + 
  geom_point() +
  labs(title = "DBSCAN Clustering Results")



# 使用K-means聚类算法将数据集分成K个类
k <- 122
kmeans_result <- kmeans(new_scale, k)
colors <- heat.colors(unique(kmeans_result$cluster))
class_mean_distances <- numeric(length(k))
class_mean_mutation_rates <- numeric(length(k))
# 创建一个列表，用于保存检验结果
test_results <- list()

# 在三维点图中可视化所有数据点和p值小于0.05的聚类中的数据点 
fig_5 <- plot_ly() %>% 
  add_trace(data = new_scale, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = "All Points") %>% 
  layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z"))) 

fig_6 <- plot_ly() %>% 
  add_trace(data = new_scale, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = "All Points") %>% 
  layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z"))) 
# 遍历每个类
i_num <- list()
j <- 1
m<-1
n<-1
mean_mutation_rate <- mean(new_scale$mutation)
higher <- list()
lower <- list()
for (i in 1:k) {
  # 将当前类作为样本1，其他类作为样本2
  group1 <- new_scale[kmeans_result$cluster == i, ]
  group2 <- new_scale[kmeans_result$cluster != i, ]
  welch_t <- t.test(group1, group2, var.equal = FALSE, alternative = "greater") # 将方向改为单侧检验，备择假设为样本1的均值大于样本2的均值
  # 将检验结果保存到列表中
  test_results[[i]] <- list(p_value = welch_t$p.value, group1 = group1)
  if (test_results[[i]]$p_value < 0.05) {
    cat("Class", i, "p-value <", 0.05, "\n")
    cat("Class", i, "is significantly different from other classes.\n")
    cat("Points in Class", i, ":\n")
    points <- which(kmeans_result$cluster == i)
    print(data[points, ]) # 输出该类中所有点的信息
    # 计算该类中所有点之间的欧几里得距离
    class_distances <- as.matrix(dist(group1))
    class_distances_vector <- as.vector(class_distances)
    class_distances_vector <- class_distances_vector[class_distances_vector != 0]
    class_mean_distances[i] <- mean(class_distances_vector)
    class_mean_mutation_rates[i] <- mean(new_scale[kmeans_result$cluster == i, "mutation"])
    # 输出该类中所有点之间距离的统计量
    cat("Class", i, "distance statistics:\n")
    cat("Mean distance:", mean(class_distances_vector), "\n")
    cat("Median distance:", median(class_distances_vector), "\n")
    cat("Minimum distance:", min(class_distances_vector), "\n")
    cat("Maximum distance:", max(class_distances_vector), "\n")
    i_num[j] <- i
    j <- j+1
    if (mean(new_scale[kmeans_result$cluster == i, "mutation"]) < mean_mutation_rate){
      lower[n] <- i
      n<-n+1
    }else{
      higher[m]<-i
      m<-m+1
    }
  }
}

# 将平均距离和平均突变率存储在一个数据框中
class_summary <- data.frame(mean_distance = class_mean_distances, mean_mutation_rate = class_mean_mutation_rates)

# 使用ggplot2包绘制折线图
library(ggplot2)
ggplot(class_summary, aes(x = mean_distance, y = mean_mutation_rate)) + geom_line() + geom_point() + xlab("Mean Distance") + ylab("Mean Mutation Rate") + ggtitle("Mean Distance vs. Mean Mutation Rate")

i_num <- data.frame(i_num)
i_num <- as.numeric(i_num)
higher<-data.frame(higher)
higher<-as.numeric(higher)
lower<-data.frame(lower)
lower<-as.numeric(lower)
for (i in 1:length(higher)){
  color = colors[i]
  cluster_points <- which(kmeans_result$cluster == higher[i])
  cluster_coordinates <- new_scale[cluster_points, c("x", "y", "z")]
  fig_5 <- fig_5 %>% add_trace(data = cluster_coordinates, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = paste("Cluster ", higher[i]), marker = list(color = color))
  
}
fig_5 <- fig_5 %>% layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z")))
# 显示散点图
fig_5

for (i in 1:length(lower)){
  color = colors[i]
  cluster_points <- which(kmeans_result$cluster == lower[i])
  cluster_coordinates <- new_scale[cluster_points, c("x", "y", "z")]
  fig_6 <- fig_6 %>% add_trace(data = cluster_coordinates, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = paste("Cluster ", lower[i]), marker = list(color = color))
  
}
fig_6 <- fig_6 %>% layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z")))
# 显示散点图
fig_6

# 初始化特征名称和p值向量
features <- c("x", "y", "z", "mutation")
p_values <- numeric(length(features))

fig_7 <- plot_ly() %>% 
  add_trace(data = new_scale, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = "All Points") %>% 
  layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z"))) 
i_num_2<-list()
q<-1
for (i in 1:length(i_num)){
  group3<-new_scale[kmeans_result$cluster == i_num[i], ]$x
  group4<-new_scale[kmeans_result$cluster != i_num[i], ]$x
  group5<-new_scale[kmeans_result$cluster == i_num[i], ]$y
  group6<-new_scale[kmeans_result$cluster != i_num[i], ]$y
  group7<-new_scale[kmeans_result$cluster == i_num[i], ]$z
  group8<-new_scale[kmeans_result$cluster != i_num[i], ]$z
  group9<-new_scale[kmeans_result$cluster == i_num[i], ]$mutation
  group10<-new_scale[kmeans_result$cluster != i_num[i], ]$mutation
  welch_t_1 <- t.test(group3, group4, var.equal = FALSE, alternative = "greater")
  welch_t_2 <- t.test(group5, group6, var.equal = FALSE, alternative = "greater")
  welch_t_3 <- t.test(group7, group8, var.equal = FALSE, alternative = "greater")
  welch_t_4 <- t.test(group9, group10, var.equal = FALSE, alternative = "greater")
  cat("类数:", i_num[i], "\n")
  cat("x特征p值：", welch_t_1$p.value,"\n")
  cat("y特征p值：", welch_t_2$p.value,"\n")
  cat("z特征p值：", welch_t_3$p.value,"\n")
  cat("mutation特征p值：", welch_t_4$p.value,"\n")
  
  # 更新p值向量
  p_values[1] <- welch_t_1$p.value
  p_values[2] <- welch_t_2$p.value
  p_values[3] <- welch_t_3$p.value
  p_values[4] <- welch_t_4$p.value
  
  # 找到最小p值对应的特征
  min_p_index <- which.min(p_values)
  min_p_feature <- features[min_p_index]
  
  # 输出结果
  cat("最小p值：", p_values[min_p_index], "\n")
  cat("最小p值对应的特征：", min_p_feature, "\n")
  
  if (features[min_p_index] == "mutation") {
    i_num_2[q] <- i_num[i]
    q<-q+1
    # 提取对应类别的数据
    p_points <- which(kmeans_result$cluster == i_num[i])
    cat("原子序数：", data$atomSerialNumber[p_points], "\n")
    cat("该类均值：", class_mean_mutation_rates[i_num[i]],"\n")
    # 提取属于当前聚类的数据点的坐标
    p_coordinates <- new_scale[p_points, c("x", "y", "z")]
    # 为聚类指定颜色
    color <- colors[i]
    # 在三维散点图中添加聚类
    fig_7 <- fig_7 %>% add_trace(data = p_coordinates, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers", name = paste("Cluster ", i_num[i]), marker = list(color = color))
  }
}
# 设定散点图的布局
fig_7 <- fig_7 %>% layout(scene = list(xaxis = list(title = "X"), yaxis = list(title = "Y"), zaxis = list(title = "Z")))
# 显示散点图
fig_7
i_num_2<-data.frame(i_num_2)
i_num_2 <- as.numeric(i_num_2)
# 提取所有显著性聚类的点的索引
significant_pointer <- which(kmeans_result$cluster %in% i_num_2)
# 提取所有显著性聚类的点的xyz和突变频率
significant_data <- new_scale[significant_pointer, c("x", "y", "z", "mutation")]
significant_sn <- data[significant_pointer,c("atomSerialNumber")]
# 输出结果
print(significant_data)
print(significant_sn)



# 将类别列添加到数据集中
new_scale$cluster <- kmeans_result$cluster

# 构建随机森林模型
rf_model <- randomForest(cluster ~ ., data = new_scale, ntree = 100)

# 查看变量的重要性
var_importance <- importance(rf_model)

# 输出变量重要性
print(var_importance)


