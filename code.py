import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Generate dataset with 50 observations and 2 variables (heights and weights)
np.random.seed(0)
heights = np.random.normal(loc=170, scale=10, size=50)
weights = np.random.normal(loc=70, scale=15, size=50)
data = pd.DataFrame({'Height': heights, 'Weight': weights})

# Descriptive statistics
descriptive_stats = data.describe()

# Mean, median, standard deviation, and range for both variables
mean_height = data['Height'].mean()
median_height = data['Height'].median()
std_height = data['Height'].std()
range_height = data['Height'].max() - data['Height'].min()

mean_weight = data['Weight'].mean()
median_weight = data['Weight'].median()
std_weight = data['Weight'].std()
range_weight = data['Weight'].max() - data['Weight'].min()

# Probability
# Define event: Probability of a person being taller than 175 cm
taller_than_175 = (data['Height'] > 175).mean()

# Distributions
# Plot histograms for both variables
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(data['Height'], bins=10, edgecolor='black')
plt.title('Height Distribution')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(data['Weight'], bins=10, color='orange', edgecolor='black')
plt.title('Weight Distribution')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Inferential Statistics - Central Limit Theorem (CLT)
sample_means = [data.sample(30).mean().values[0] for _ in range(1000)]

# Plot distribution of sample means
plt.figure(figsize=(6, 4))
plt.hist(sample_means, bins=20, edgecolor='black')
plt.title('Distribution of Sample Means (n=30)')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.show()

# Confidence Intervals (CI)
# Calculate 95% confidence interval for the mean height
ci_height = stats.norm.interval(0.95, loc=mean_height, scale=std_height/np.sqrt(len(data)))

# Hypothesis Testing
# Formulate hypothesis: Testing if mean height is different from 170 cm
# Null Hypothesis: Mean height = 170 cm
# Alternative Hypothesis: Mean height ≠ 170 cm
t_stat, p_value = stats.ttest_1samp(data['Height'], 170)
alpha = 0.05
if p_value < alpha:
    hypothesis_result = "Reject null hypothesis"
else:
    hypothesis_result = "Fail to reject null hypothesis"

# Critical Regions, Level of Significance, and Error Types
# For a two-tailed test at alpha = 0.05, critical region is outside the range (-1.96, 1.96)

# Choosing P-values from Tables
# Given significance level alpha = 0.05, find critical value from t-table or z-table
critical_value = stats.t.ppf(1 - alpha/2, df=len(data)-1)

# Feature Selection Using P-values
# Perform hypothesis test to determine if height variable is a significant predictor
t_stat_height, p_value_height = stats.ttest_ind(data['Height'], data['Weight'])

if p_value_height < alpha:
    feature_selection_result = "Include height variable"
else:
    feature_selection_result = "Exclude height variable"

print("Descriptive Statistics:")
print(descriptive_stats)
print("\nMean, Median, Standard Deviation, and Range for Height:")
print("Mean:", mean_height)
print("Median:", median_height)
print("Standard Deviation:", std_height)
print("Range:", range_height)
print("\nMean, Median, Standard Deviation, and Range for Weight:")
print("Mean:", mean_weight)
print("Median:", median_weight)
print("Standard Deviation:", std_weight)
print("Range:", range_weight)
print("\nProbability of a person being taller than 175 cm:", taller_than_175)
print("\n95% Confidence Interval for the mean height:", ci_height)
print("\nHypothesis Testing Result for Mean Height ≠ 170 cm:")
print("T-statistic:", t_stat)
print("P-value:", p_value)
print("Decision:", hypothesis_result)
print("\nCritical Value for a two-tailed test at alpha=0.05:", critical_value)
print("\nFeature Selection Result for Height Variable:")
print(feature_selection_result)
