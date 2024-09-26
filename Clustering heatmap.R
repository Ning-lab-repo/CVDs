library(ConsensusClusterPlus)
library(cluster)
library(factoextra)
library(pheatmap)
library(tibble)
library(dplyr)

# 列出当前加载的包
search()

# 设置参数
file_path <- "特征data.csv"
data_matrix <- read.csv(file_path, row.names = 1, check.names = F)
file_path <- "分簇data.csv"
Cluster <- read.csv(file_path, row.names = 1, check.names = F)

# 确保数据是数值矩阵
d <- as.matrix(data_matrix)


# 对每个特征进行标准化的函数
standardize <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

# 对每个特征应用标准化函数
d_standardized <- apply(d, 1, standardize)

# 将标准化后的值缩放到-3到3的范围的函数
scale_to_range <- function(x, min_val, max_val) {
  scaled_x <- (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  scaled_x <- scaled_x * (max_val - min_val) + min_val
  return(scaled_x)
}

# 设置缩放范围
min_val <- -3
max_val <- 3

# 对每个特征应用缩放函数
d <- apply(d_standardized, 1, scale_to_range, min_val = min_val, max_val = max_val)


# 显示每个聚类的样本数
table(Cluster$Cluster)
#Cluster$Cluster <- ifelse(Cluster$Cluster == "1", "C1", 'C2')
Cluster$Cluster <- ifelse(Cluster$Cluster == "1", "C1",
                          ifelse(Cluster$Cluster == "2", "C2",
                                 ifelse(Cluster$Cluster == "3", "C3",NA)))
Cluster$Cluster <- factor(Cluster$Cluster, levels = c("C1", "C2","C3"))
save(Cluster, file = "Cluster.Rda")
# 将行名转换为列名
Cluster <- rownames_to_column(Cluster, "sample")

# 按照Cluster列排序，并将样本信息设为行名
annotation <- Cluster %>% 
  arrange(Cluster) %>% 
  column_to_rownames(var = "sample")

# 确保Cluster数据框的列名设置正确
if (ncol(Cluster) != 2) {
  stop("Cluster数据框应包含2列：sample和Cluster")
}

# 按Cluster排序
a <- Cluster %>% arrange(Cluster)
# 获取样本的顺序
sample_order <- match(a$sample, colnames(d))

# 按照样本顺序重新排序数据框 d
c <- d[, sample_order]

# 热图绘制
ann_colors <- list(Cluster = c(C1 = "#003399", C2 = "red2", C3 = "green"))  # 设置颜色
pheatmap(c, annotation_col = annotation, annotation_colors = ann_colors, 
         color = colorRampPalette(c("#3399FF", "white", "firebrick3"))(length(seq(-3, 3, by = 0.001))),
         breaks = seq(-3, 3, by = 0.001), 
         legend_breaks = seq(-3, 3, 1), 
         cluster_cols = F, 
         cluster_rows = T, 
         show_rownames = T,
         fontsize =16, 
         fontsize_row = 14,
         gaps_col = NULL,           # 可在列间添加空隙
         gaps_row = NULL,           # 可在行间添加空隙
         main = "Feature clustering heatmap of three clusters",
         fontsize_col = 5, 
         show_colnames = F)

