library(ConsensusClusterPlus)
library(cluster)
library(factoextra)
library(pheatmap)
# 列出当前加载的包
search()
# 设置参数
file_path <- "data.csv"
data_matrix <- read.csv(file_path)

data_matrix <-data_matrix[,-c (1)]
# 确保数据是数值矩阵
d <- as.matrix(data_matrix)

d = sweep(d,1, apply(d,1,median,na.rm=T))
df<- t(d)

max_k <- 6  # 最大的k值
reps <- 1000  # 重复次数，通常设置为100或者更多
p_item <- 0.8  # 选择样本的比例
p_feature <- 1  # 选择特征的比例，通常设置为1

#运行ConsensusClusterPlus
results <- ConsensusClusterPlus(d,
                                maxK = max_k,
                                reps = reps,
                                pItem = p_item,
                                pFeature = p_feature,
                                title = "Consensus Clustering",
                                clusterAlg = "km",  # 可以选择其他聚类算法，例如 "km" (k-means)
                                distance = "euclidean",  # 选择距离度量方法，例如 "euclidean"
                                seed = 123,  # 设定种子以保证结果可重复
                                plot = "pdf")  # 输出图表格式，例如 "png"
k <- 3
cluster_assignments <- results[[k]]$consensusClass

# 获取样本ID
sample_ids <- colnames(d)

# 创建一个数据框来存储样本ID及其对应的簇
cluster_data <- data.frame(SampleID = sample_ids, Cluster = cluster_assignments)

# 按簇分组并将每个簇的样本ID存储到不同的CSV文件中
for (cluster in unique(cluster_data$Cluster)) {
  cluster_samples <- cluster_data[cluster_data$Cluster == cluster, "SampleID"]
  write.csv(cluster_samples, file = paste0("cluster_", cluster, ".csv"), row.names = FALSE, col.names = FALSE)
}

print("每个簇的样本ID已成功存储到CSV文件中")
# 提取共识矩阵并转换为整数向量
consensus_matrix_k4 <- as.integer(results[[4]]$consensusClass)
consensus_matrix_k3 <- as.integer(results[[3]]$consensusClass)

# 计算不相似矩阵

dist_matrix <- dist(df)
#计算轮廓宽度
silhouette_k4 <- silhouette(consensus_matrix_k4, dist_matrix)
silhouette_k3 <- silhouette(consensus_matrix_k3, dist_matrix)

pdf("3 and 4 silhouette_plots.pdf", width = 10, height = 5)  # 调整宽度和高度
# 设置所有字体大小为16磅
par(mfrow = c(1, 2), cex.axis = 1.33, cex.lab = 1.33, cex.main = 1.33, cex = 1.2)


# k=3的轮廓图
plot(silhouette_k3, col = rainbow(3), border = NA, main = "Silhouette plot for k=3")
# 添加平均轮廓宽度线
abline(v = mean(silhouette_k3[, 'sil_width']), lty = 2, col = "black")

#k=4的轮廓图
plot(silhouette_k4, col = rainbow(4), border = NA, main = "Silhouette plot for k=4")
# 添加平均轮廓宽度线
abline(v = mean(silhouette_k4[, 'sil_width']), lty = 2, col = "black")



dev.off()

print("轮廓图已成功保存为PDF文件")