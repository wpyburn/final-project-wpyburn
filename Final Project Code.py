import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read data from file
data = np.loadtxt('data.txt', delimiter=',')
years = data[:, 0]
epoch_change = data[:, 1]

# Create the plot
plt.figure(figsize=(12, 10))

# First subplot - Original plot
plt.subplot(2, 1, 1)
ax1 = plt.gca()

# Plot data points
plt.scatter(years, epoch_change, color='blue', s=50, label='Observed Data')

# Create x values for the theoretical curve
x_theory = np.linspace(1973, 2005, 1000)

# Calculate theoretical prediction
constant = (-2.40242e-12)/(2*27898.56)
time_factor = 3.154e7
y_theory = constant * (time_factor * (x_theory - 1975))**2

# Plot theoretical curve
plt.plot(x_theory, y_theory, 'r-', label='General Relativity Prediction')

# Set axis limits
plt.xlim(1973, 2005)
plt.ylim(-40, 5)

# Customize axis labels and title
fnt = 16
ax1.set_xlabel('Year', fontsize=fnt)
ax1.set_ylabel('Cumulative Shift of periastron time (s)', fontsize=fnt)
ax1.set_title('Orbital Decay of PSR B1913+16', fontsize=fnt)

# Customize ticks
ax1.minorticks_on()
ax1.tick_params(which='both', axis='both', direction='in', top=True, right=True, labelsize=14)
ax1.tick_params(which='major', length=10)
ax1.tick_params(which='minor', length=5)

# Add legend
plt.legend(fontsize=12)

# Add grid 
plt.grid(True, linestyle='--', alpha=0.7)

# Second subplot - Differences
plt.subplot(2, 1, 2)
ax2 = plt.gca()

# Calculate differences between observed and theoretical data
theoretical_at_observed = np.interp(years, x_theory, y_theory)
differences = epoch_change - theoretical_at_observed

# Perform linear regression on the differences
slope, intercept, r_value, p_value, std_err = stats.linregress(years, differences)
regression_line = slope * years + intercept

# Plot differences
plt.scatter(years, differences, color='green', s=50, label='Observed - Theoretical')
plt.plot(years, regression_line, 'r--', label=f'Linear fit (slope={slope:.2e})')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Add a reference line at y=0

# Customize the difference plot
ax2.set_xlabel('Year', fontsize=fnt)
ax2.set_ylabel('Difference (s)', fontsize=fnt)
ax2.set_title('Difference between Observed and Theoretical Values', fontsize=fnt)
ax2.minorticks_on()
ax2.tick_params(which='both', axis='both', direction='in', top=True, right=True, labelsize=14)
ax2.tick_params(which='major', length=10)
ax2.tick_params(which='minor', length=5)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(1973, 2005)
# Adjust layout
plt.tight_layout()

# Create and display the comparison table
print("\nComparison between Observed and Theoretical Values:")
print("Year    Observed    Theoretical    Difference")
print("-" * 45)
for year, obs, theo, diff in zip(years, epoch_change, theoretical_at_observed, differences):
    print(f"{year:.1f}    {obs:9.3f}    {theo:11.3f}    {diff:9.3f}")

# Add statistical information
print("\nLinear Regression Analysis of Differences:")
print(f"Slope: {slope:.2e} seconds/year")
print(f"Intercept: {intercept:.2e} seconds")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Standard Error: {std_err:.2e}")

# Calculate statistical tests
mean_diff = np.mean(differences)
std_diff = np.std(differences)
std_error = std_diff / np.sqrt(len(differences))

# One-sample t-test
t_stat, t_p_value = stats.ttest_1samp(differences, 0)

# Shapiro-Wilk test (tests for Gaussian distribution)
shapiro_stat, shapiro_p = stats.shapiro(differences)

# Calculate 95% confidence interval
confidence = 0.95
degrees_of_freedom = len(differences) - 1
ci = stats.t.interval(confidence=confidence, 
                     df=degrees_of_freedom,
                     loc=mean_diff,
                     scale=std_error)

# Print statistical analysis results
print("\nStatistical Analysis of Differences:")
print("-" * 50)
print(f"Basic Statistics:")
print(f"Mean difference: {mean_diff:.3f} ± {std_error:.3f} seconds")
print(f"Standard deviation: {std_diff:.3f} seconds")

print("\nGaussian Distribution Test (Shapiro-Wilk):")
print(f"W-statistic: {shapiro_stat:.3f}")
print(f"p-value: {shapiro_p:.4f}")
print(f"{'Data follows Gaussian distribution' if shapiro_p > 0.05 else 'Data deviates from Gaussian distribution'}")

print("\nOne-sample t-test (H0: mean difference = 0):")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {t_p_value:.4f}")
print(f"{'Evidence of systematic deviation' if t_p_value < 0.05 else 'No evidence of systematic deviation'}")

print(f"\n95% Confidence Interval: ({ci[0]:.3f}, {ci[1]:.3f}) seconds")

plt.show()
