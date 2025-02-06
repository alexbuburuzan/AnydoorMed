import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# Update matplotlib configuration to use Helvetica font
plt.rcParams.update({
    "text.usetex": True,  # Disable LaTeX rendering
    # "font.family": "Helvetica"
})

# Load the dataset
df = pd.read_csv('/mnt/iusers01/fse-ugpgt01/compsci01/m18453ab/project/AnyDoor-Med/data/vindr-mammo/finding_annotations.csv') 

# Convert finding_categories from string representation of lists to actual lists
df['finding_categories'] = df['finding_categories'].apply(ast.literal_eval)

# Split dataset into training and testing
df_train = df[df['split'] == 'training']
df_test = df[df['split'] == 'test']

# Flatten the list of findings and count frequencies
from collections import Counter
all_findings_train = [finding for sublist in df_train['finding_categories'] for finding in sublist]
all_findings_test = [finding for sublist in df_test['finding_categories'] for finding in sublist]

finding_counts_train = Counter(all_findings_train)
finding_counts_test = Counter(all_findings_test)

del finding_counts_train['No Finding']
del finding_counts_test['No Finding']

# Convert the Counter dictionary to a DataFrame for plotting
finding_df_train = pd.DataFrame(list(finding_counts_train.items()), columns=['Finding', 'Count']).sort_values(by='Finding')
finding_df_test = pd.DataFrame(list(finding_counts_test.items()), columns=['Finding', 'Count']).sort_values(by='Finding')

# Plot frequency of finding categories
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(data=finding_df_train, x='Count', y='Finding', palette='magma', ax=axes[0])
sns.barplot(data=finding_df_test, x='Count', y='Finding', palette='magma', ax=axes[1])
axes[0].set_title('Training Set - Frequency of Finding Categories')
axes[1].set_title('Testing Set - Frequency of Finding Categories')
axes[1].set_ylabel('')  # Remove y-axis label for the second subplot
plt.tight_layout()
plt.savefig('visualisations/finding_categories_histogram.pdf', format="pdf")
plt.close()

# Plot histogram of BI-RADS assessments
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
birads_order = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]
sns.countplot(data=df_train, x='breast_birads', palette='coolwarm', ax=axes[0], order=birads_order)
sns.countplot(data=df_test, x='breast_birads', palette='coolwarm', ax=axes[1], order=birads_order)
axes[0].set_title('Training Set - Histogram of BI-RADS Assessments')
axes[1].set_title('Testing Set - Histogram of BI-RADS Assessments')
axes[0].set_xlabel('BI-RADS score')
axes[1].set_xlabel('BI-RADS score')
axes[1].set_ylabel('')  # Remove y-axis label for the second subplot
plt.tight_layout()
plt.savefig('visualisations/birads_histogram.pdf', format="pdf")
plt.close()

print("Plots saved as finding_categories_histogram.png, birads_histogram.png, and age_histogram.png")
