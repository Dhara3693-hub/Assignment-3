# %% read data
import pandas as pd

df = pd.read_csv("seeds_dataset.txt", sep="\t+", header=None)
# 1 area A,
# 2 perimeter P,
# 3 compactness C = 4*pi*A/P^2,
# 4 length of kernel,
# 5 width of kernel,
# 6 asymmetry coefficient
# 7 length of kernel groove.
# 8 target
df.columns = [
    "area",
    "perimeter",
    "compactness",
    "length_kernel",
    "width_kernel",
    "asymmetry_coefficient",
    "length_kernel_groove",
    "target",
]


# %%
df.describe()


#%%
import seaborn as sns


sns.scatterplot(
    x="area",
    y="asymmetry_coefficient",
    data=df,
    hue="target",
    legend="full",
)


# %% also try lmplot and pairplot

#  Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#  Read data
df = pd.read_csv("seeds_dataset.txt", sep="\t+", header=None)
# Column names
df.columns = [
    "area",
    "perimeter",
    "compactness",
    "length_kernel",
    "width_kernel",
    "asymmetry_coefficient",
    "length_kernel_groove",
    "target",
]

#  Describe the data
print(df.describe())

#  Scatter plot of area vs asymmetry coefficient colored by target
sns.scatterplot(
    x="area",
    y="asymmetry_coefficient",
    data=df,
    hue="target",
    legend="full",
)
plt.title('Scatter Plot of Area vs Asymmetry Coefficient')
plt.show()

#  lmplot of area vs asymmetry coefficient with target hues
sns.lmplot(
    data=df,
    x='area',
    y='asymmetry_coefficient',
    hue='target',
    markers=["o", "s", "D"],
    palette='viridis',
    height=6
)
plt.title('lmplot of Area vs Asymmetry Coefficient by Target')
plt.show()

#  pairplot of all features colored by target
sns.pairplot(df, hue='target', palette='viridis')
plt.suptitle('Pairplot of All Features by Target', y=1.02)
plt.show()


# %%

#  Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score

#  Read data
df = pd.read_csv("seeds_dataset.txt", sep="\t+", header=None)
# Column names
df.columns = [
    "area",
    "perimeter",
    "compactness",
    "length_kernel",
    "width_kernel",
    "asymmetry_coefficient",
    "length_kernel_groove",
    "target",
]

# Determine the best number of clusters
x = df.drop("target", axis=1)  # Features
y = df["target"]                # Target
inertia = {}
homogeneity = {}

# Loop over candidate numbers of clusters
for n_clusters in range(2, 11):  # Example range from 2 to 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(x)  # Fit the model
    inertia[n_clusters] = kmeans.inertia_  # Store inertia
    y_kmeans = kmeans.predict(x)  # Predict cluster labels
    homogeneity[n_clusters] = homogeneity_score(y, y_kmeans)  # Store homogeneity score

#  Plotting inertia and homogeneity scores
plt.figure(figsize=(12, 6))

# Inertia
ax = sns.lineplot(
    x=list(inertia.keys()),
    y=list(inertia.values()),
    color="blue",
    label="Inertia",
    legend=None,
)
ax.set_ylabel("Inertia")
ax.set_xlabel("Number of Clusters")
ax.set_title("Inertia and Homogeneity Scores")

# Homogeneity
ax2 = ax.twinx()
ax2 = sns.lineplot(
    x=list(homogeneity.keys()),
    y=list(homogeneity.values()),
    color="red",
    label="Homogeneity",
    legend=None,
)
ax2.set_ylabel("Homogeneity")

# Adding legend
ax.figure.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))

plt.show()

# %%
