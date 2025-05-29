library(forestploter)
library(grid)

result2 <- read.csv("x.csv")
# 把NA替换为空字符串
#result2$`Number (%)` <- ifelse(is.na(result2$`Number (%)`), "", result2$`Number (%)`)

# 计算标准误差（SE），它在绘图的时候会表示正方形的大小
result2$se <- (log(result2$upper.95) - log(result2$HR))/1.96

# 为森林图添加空白列，为了产生一个绘图区间，用于显示CI
result2$` ` <- paste(rep(" ", 15), collapse = " ")
# 定义一个简单的主题，大家可以随意发挥自己的审美！
tm <- forest_theme(base_size = 15,           # 设置基础字体大小
                   refline_gp = gpar(col = "red4"),     # 设置参考线颜色为红色
                   arrow_type = "closed",    # 设置箭头类型为闭合箭头
                   footnote_gp = gpar(col = "blue4"),
                   font_gp = gpar(fontfamily = "Arial"))  # 使用支持特殊字符的字体)   # 设置脚注文字颜色为蓝色

# 绘制森林图
p <- forest(result2[,c(1, 2,8,5,6)],   # 选择要在森林图中使用的数据列，这里包括变量名列、患者数量列、绘图要用的空白列和HR（95%CI）列
            est = result2$HR,          # 效应值，也就是HR列
            lower = result2$lower.95,  # 置信区间下限
            upper = result2$upper.95,  # 置信区间上限
            sizes = 2*result2$se,        # 黑框框的大小
            ci_column = 3,             # 在第3列（可信区间列）绘制森林图
            ref_line = 1,              # 添加参考线
            arrow_lab = c("Low risk", "High Risk"),  # 箭头标签，用来表示效应方向，如何设置取决于你的样本情况
            xlim = c(0.7, 2.2),          # 设置x轴范围
            ticks_at = c(0.7,1,1.3,1.6,1.9,2.2),  # 在指定位置添加刻度
            theme = tm,                # 添加自定义主题
            line_size = 2.5,
            column_labels = c("Group", "HR", " ", "95% CI", "P-value"))  # 添加脚注信息
print(p)
# 设置保存路径和文件名
pdf("x.pdf", width = 12, height = 8)

# 绘制森林图（你刚才画好的 p）
grid::grid.draw(p)

# 关闭设备
dev.off()
