# 加载必要的库
library(survival)
library(survminer)

# 读取数据
Cluster_res <- read.csv("data.csv") # 请替换为你的数据文件路径
# 计算p值
fitd=survdiff(Surv(time,status)~results,#作与group相关联的生存分析
              data=Cluster_res,#用surv的表格数据去做
              na.action=na.exclude)#固定默认
pValue=1-pchisq(fitd$chisq,length(fitd$n)-1)#计算p值

# 进行生存分析
fit <- survfit(Surv(time, status) ~ results, data = Cluster_res)


summary(fit)
p.lab=paste0("p",ifelse(pValue<0.001,"<0.001",paste0("=",round(pValue,3))))#
# 绘制生存曲线
surv_plot <- ggsurvplot(
  fit, 
  data = Cluster_res,
  pval = p.lab,
  conf.int = F,#显示置信区间
  palette = "lancet",
  legend = c(0.9, 0.25), # 通过坐标指定图例位置
  legend.title = "Cluster", 
  legend.labs = c("1", "2","3"), # 更改图例标题和标签
  surv.median.line = "hv",
  size=1,#规定线条粗细
  xlab="Time(months)",xlim=c(0,240),break.time.by=40,#设置X轴长度，每步步长为40
  ylab="Survival probability(%)",
  risk.table = TRUE,risk.table.col="strata",
  risk.table.y.text.col = TRUE, 
  tables.height = 0.3,
  ggtheme = theme_classic() +  # 使用 classic 主题，去除背景和边框
    theme(
      axis.title.x = element_text(size = 16), # 修改 X 轴标签字体大小为 16 磅
      axis.title.y = element_text(size = 16), # 修改 Y 轴标签字体大小为 16 磅
      axis.text.x = element_text(size = 16),  # 修改 X 轴刻度字体大小为 16 磅
      axis.text.y = element_text(size = 16),  # 修改 Y 轴刻度字体大小为 16 磅
      panel.grid.major = element_blank(),     # 去除主背景网格线
      panel.grid.minor = element_blank(),     # 去除次背景网格线
      panel.border = element_blank(),         # 去除面板边框
      axis.line = element_line(size = 0.6)    # 添加X和Y轴线
    )
)

# 或者保存为PDF文件
pdf("生存曲线.pdf", width = 8, height = 10)
print(surv_plot)
dev.off()