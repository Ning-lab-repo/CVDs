# CVDs
# Note  
Cardiovascular and cerebrovascular diseases are major global public health issues, contributing significantly to the burden of disease worldwide. Deaths from cardiovascular and cerebrovascular diseases account for a large proportion of global mortality. However, among those who die from these diseases, the pathophysiological characteristics they exhibit may vary. Therefore, we primarily explore the differences among individuals who also die from cardiovascular and cerebrovascular diseases.
# The description of codes  
  Clustering heatmap.R  This code generates a feature clustering heatmap by grouping, where each column represents a sample and each row represents a feature.  
  Silhouette plot.R  This code includes consensus clustering and calculates silhouette coefficients using a consensus matrix to create silhouette plots. The results shown in these plots can help determine the optimal k value, which corresponds to the number of clusters.  
  Survival curve.R  This code is used to plot survival curves and risk tables for k clusters of populations, and it includes code for conducting differential tests on the survival curves.  
  Bar plot.py  For categorical variables, a bar chart is used to clearly display the differences in proportions of various categories between different groups, and it includes differential testing among multiple groups.  
  Boxplot.py  Continuous variables are represented using box plots, along with conducting differential analysis.  
  Data extraction and merging.py  This code is used to extract data from the database by specifying the names of certain columns needed, and also to merge the data based on column names or id.  
  Define variable.py  This code is used to redefine variables based on several conditions. For example, the education level indicator is defined as 1 for elementary or middle school and as 2 for high school.  
  Model training.py  This code is used for training a DNN model, including handling imbalanced sample distribution using SMOTE, selecting the optimal parameters for the model, plotting the ROC curve, and extracting the results from the second-to-last layer of the DNN.  
  Multiple imputation.py  This code is used for performing multiple imputation on missing data, similar to the MICE package in R.  
  Single feature ROC.py  This code is used to plot the ROC curves for individual features distinguishing multiple classification outcomes, and it combines the ROC curves of multiple single indicators into one plot.  
  UMAP.py  This code performs UMAP dimensionality reduction on samples, visualizing the distribution of the population in a lower-dimensional space, and allows for mapping feature values based on ID numbers.
# The description of CVDs
In summary, it was found that populations who died from cardiovascular diseases exhibit different survival conditions and characteristics.
