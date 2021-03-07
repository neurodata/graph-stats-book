# Learning Graph Representations

## Network and Vertex feature joint representation learning

In many network problems, you might have access to much more information than just the 
collection of nodes and edges in the network. If you were investigating a social 
network, for instance, you might have access to extra information about each 
person -- their gender, for instance, or their age. When we you embed a network, it seems 
like you should be able to use this information - called "features" or "covariates" - to somehow improve your analysis.
Many of the techniques and tools that we'll explore in this section use both the covariates and the network to 
learn from new, holistic representations of the data available to us, jointly using both the network and the covariates.
These techniques are called joint representation learning.

There are two primary reasons that we might want to explore using node covariates in addition to networks. Firstly, they might improve our 
standard embedding algorithms, like Laplacian and Adjacency Spectral Embedding. 
For example, if the latent structure of the covariates lines up with the latent structure
of our network, then we could conceivably reduce noise, even if they don't overlap perfectly. Second,
figuring out what the clusters of an embedding actually mean can sometimes be difficult, and using covariates
gives us to access to a natural structure. If we're clustering brain networks, for instance, 
covariate information telling us a location in the brain and name of brain region for each node might let us better
cluster by region.

### Generating Data

We need a graph and some covariates. To start off, we'll make a pretty straightforward Stochastic Block Model with 1500 nodes and 3 communities.

#### Stochastic Block Model

import numpy as np
from graspologic.simulations import sbm
from graspologic.plot import heatmap
import warnings
warnings.filterwarnings("ignore")  # TODO: don't do this, fix scatterplot

# Start with some simple parameters
N = 1500
n = N // 3
p, q = .3, .15
B = np.array([[p, q, q],
              [q, p, q],
              [q, q, p]])

# Make our Stochastic Block Model
A, labels = sbm([n, n, n], B, return_labels = True)
heatmap(A, title="Our Stochastic Block Model");

#### Covariates

Now, let's generate some covariates. Remember, each node is associated with its own group of covariates that provide information about the node. We'll organize these into a matrix, where each row contains the covariates associated with a particular node.  

To keep things simple, we'll have our covariates only take on true/false values - or, more specifically, 0 and 1. We'd also like a node's covariates to look different depending on which community it belongs to. To that end, we'll give each node 30 covariates, with the first 10 having a higher probability of 1 in the first community, the second having a higher probability of 1 in the second community, and the third having a higher probability of 1 in the third community.

a = {"a": 1}
b = {"b": 2}

a.update(b)
a

a["c"] = 3
a

import pandas as pd

df1 = pd.DataFrame(dict(x=[1,2,3], y="str"))
df2 = pd.DataFrame(dict(x=[4,5,6], y="str"))

pd.concat((df1, df2)).reset_index(drop=True)

import numpy as np
from scipy.stats import bernoulli
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def gen_covariates(p1=.8, p2=.3, N=1500):
    """
    Generate a matrix of covariates.
    """
    n_covariates = 30

    bern = lambda p: bernoulli.rvs(p, size=(N//3, n_covariates//3))    
    X = np.block([[bern(p1), bern(p2), bern(p2)],
                  [bern(p2), bern(p1), bern(p2)],
                  [bern(p2), bern(p2), bern(p1)]])

    return X

# Generate a covariate matrix
X = gen_covariates(N=N)

# Plot and make the axis look nice
fig, ax = plt.subplots(figsize=(5, 8))
ax = sns.heatmap(X, cmap=cmap, ax=ax)
ax.set(title="Visualization of the covariates", xticks=[], 
       ylabel="Rows containing covariates for their respective nodes");

# make the colorbar look nice
colors = ["white", "black"]
cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
cbar = ax.collections[0].colorbar
cbar.set_ticks([0.25, 0.75])
cbar.set_ticklabels(['0', '1'])
cbar.ax.set_frame_on(True)

## Covariate-Assisted Spectral Embedding

<i>Covariate-Assisted Spectral Embedding</i>, or CASE<sup>1</sup>, is a simple way of combining our graph and our covariates into a single model. In the most straightforward version of CASE, we use the graph's regularized Laplacian matrix $L$ and a function of our covariate matrix $XX^T$ (where $X$ is just our covariate matrix, in which row $i$ contains the covariates associated with node $i$). Notice the word "regularized" - This means (from the Laplacian section earlier) that we are using $L_{\tau} = D_{\tau}^{-1/2} A D_{\tau}^{-1/2}$.

Remember that, in our case, $X$ only contains 0's and 1's. To interpret $XX^T$, remember from linear algebra that we're taking the weighted sum (or, in math parlance, the dot product) of each row with each other row, as shown below:

\begin{align}
\begin{bmatrix}
1 \\
1 \\
1 \\
\end{bmatrix} \cdot 
\begin{bmatrix}
0 \\
1 \\
1 \\
\end{bmatrix} = 1\times 0 + 1\times 1 + 1\times 1 = 2
\end{align}

If there are two overlapping 1's in the same place in rows $i$ and $j$, then there will be a 1 in that place in the weighted sum. The resulting value, $XX^T_{i, j}$ will be equal to the number of positions in which rows $i$ and $j$ both have ones. 

**TODO**: figure here --> zoom in on two rows of the covariate matrix figure above, show dot product visually, show with a color that the result is larger depending on the overlap

So, a position of $XX^T$ can be interpreted as measuring the "agreement" between rows $i$ and row $j$. The result is a matrix that looks fairly similar to our Laplacian!  

The following Python code generates both matrices and visualizes them.

from graspologic.utils import to_laplacian
import matplotlib.pyplot as plt

L = to_laplacian(A, form="R-DAD")
XXt = X@X.T

# plot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5), constrained_layout=True)
L_ax = heatmap(L, title=r"Regularized Laplacian", ax=axs[0])
X_ax = heatmap(XXt, title="Covariate matrix times its transpose", ax=axs[1]);

CASE simply sums these two matrices together, using a weight for $XX^T$ so that they both contribute equally to the result. Here, we'll just use the ratio of the leading eigenvalues of our two matrices as our weight (henceforth known as $\alpha$). Later on, we'll explore ways to pick a better $\alpha$.

# Get a simple weight
L_eigvals = np.flip(np.linalg.eigvalsh(L))
XXt_eigvals = np.flip(np.linalg.eigvalsh(XXt))

# Ratio of the leading eigenvalues of L and XX^T
alpha = np.float(L_eigvals[0] / XXt_eigvals[0])

# Using our simple weight, combine our two matrices
L_ = L + alpha * X@X.T
heatmap(L_, title="Our Combined Laplacian and covariates matrix");

Everything works as usual from here: we decompose our matrix using an SVD, extracting the first two eigenvectors, and then we plot the rows to visualize our clustering.

from graspologic.embed import selectSVD
from graspologic.plot import pairplot
import scipy

def plot_latents(latent_positions, *, title, labels):
    plot = sns.scatterplot(latent_positions[:, 1], latent_positions[:, 2], 
                           hue=labels, palette="Set1", linewidth=0, s=10)
    plot.set_title(title);

def embed(A, *, n_clusters):
    latents, _, _ = scipy.linalg.svd(A)
    latents = latents[:, :n_clusters]
    return latents

latents = embed(L_, n_clusters=3, labels=labels)
plot_latents(latents, title="Embedding our model")

### Setting A Better Weight

Our simple choice of the ratio of leading eigenvalues for our weight $\alpha$ is straightforward, but we can probably do better. If our covariate matrix is crappy -- meaning, it doesn't tell us much about our communities -- we'd want a smaller weight so that our Laplacian is more emphasized when we embed. Similarly, if our Laplacian is crappy, we'd like a larger weight to emphasize the covariates.  

In general, we'd simply like to embed in a way that makes our clustering better - meaning, if we label our communities, we'd like to be able to correctly retrieve as many labels after the embedding as possible with a clustering algorithm, and for our clusters to be as distinct as possible.

One reasonable way to accomplish this goal is to simply find a range of possible $\alpha$ values, embed for every value in this range, and then to simply check which values produce the best clustering.

#### Getting the Range

For somewhat complicated linear algebra reasons<sup>1</sup>, it's fairly straightforward to get a range of tuning values: the minimum and maximum $\alpha$ is described by a set of two equations.

$\alpha_{min} = \frac{\lambda_K(L) - \lambda_{K+1}(L)}{\lambda_1(XX^T)}$ where $K$ is the number of embedding clusters, $\lambda_i(L)$ is the $i_{th}$ eigenvalue of $L$.

If the number of covariate dimensions is less than or equal to the number of clusters, then  
$\alpha_{max} = \frac{\lambda_1 (L)}{\lambda_R (XX^T)}$ where $R$ is the number of covariate dimensions

If the number of covariate dimensions is greater than the number of clusters, then  
$\alpha_{max} = \frac{\lambda_1(L)}{\lambda_K(XX^T) -\lambda_{K+1} (XX^T)}$

R = X.shape[1]  # Number of covariates
K = B.shape[0]  # Number of clusters

# Remember, Python uses 0-indexing!
amin = (L_eigvals[K-1] - L_eigvals[K]) / XXt_eigvals[0]
if R <= K:
    amax = L_eigvals[0] / XXt_eigvals[R-1]
elif R > K:
    amax = L_eigvals[0] / (XXt_eigvals[K-1] - XXt_eigvals[K])
    
print(f"Our minimum weight is {amin:.8f}.")
print(f"Our maximum weight is {amax:.8f}.")

#### K-Means

Let's try it out. Our clustering algorithm of choice will be scikit-learn's faster implementation of k-means. The K-means algorithm is a simple algorithm capable of clustering most datasets very quickly and efficiently, often in only a few iterations. It works by at first putting cluster centers in essentially random places, then iterating through a center-finding procedure until the cluster centers are in nice places. If you want more information, you can check out the original paper by Stuart Lloyd<sup>2</sup>.

We also need to define exactly what it means to check which values produce the best clustering. Fortunately, K-means comes out-of-the-box with exactly what we need: its objective function, the sum of squared distances of each point from its center. In KMeans, this is generally called the "inertia".

from sklearn.cluster import MiniBatchKMeans
import time

def cluster(alpha, L, XXt, *, n_clusters):
    L_ = L + alpha*XXt
    latents, _, _ = scipy.linalg.svd(L_)
    latents = latents[:, :n_clusters]
    return latents


start = time.perf_counter()
tuning_range = np.linspace(amin, amax, num=10)
alphas = {}
for alpha in tuning_range:
    L_ = L + alpha*XXt
    latents = embed(L_, n_clusters=K)
    kmeans = MiniBatchKMeans(n_clusters=K)
    kmeans.fit(latents)
    alphas[alpha] = kmeans.inertia_

best_alpha = min(alphas, key=alphas.get)
print(f"Our best alpha-value after tuning is {best_alpha:0.8f}")

latents = embed(L+best_alpha*XXt, n_clusters=K)
plot_latents(latents, title="Our embedding after tuning", labels=labels)

heatmap(L+best_alpha*XXt)

### Variations on CASE

There are situations where changing the matrix that you embed is useful. 

*non-assortative*  
If your graph is *non-assortative* - meaning, the between-block probabilities are greater than the within-block communities - it's better to square your Laplacian. You end up embedding $LL + aXX^T$.  

*big graphs*  
Since the tuning procedure is computationally expensive, you wouldn't want to spend the time tuning $\alpha$ for larger graphs. There are a few options here: you can use a non-tuned version of alpha, or you can use a variant on classical correlation analysis<sup>3</sup> and simply embed $LX$.

### Using Graspologic

Graspologic's CovariateAssistedSpectralEmbedding class uses SVD decomposition to implement CASE, just like we just did earlier. The following code applies CASE to reduce the dimensionality of $L + aXX^T$ down to three dimensions, and then plots the second dimension against the third to show the clustering.

from graspologic.embed import CovariateAssistedEmbedding as CASE

casc = CASE(embedding_alg="assortative", n_components=3)
latents = casc.fit_transform(A, covariates=X)
plot_latents(latents, "Embedding our model using graspologic")

#### References

[1] N. Binkiewicz, J. T. Vogelstein, K. Rohe, Covariate-assisted spectral clustering, Biometrika, Volume 104, Issue 2, June 2017, Pages 361–377, https://doi.org/10.1093/biomet/asx008  
[2] Lloyd, S. (1982). Least squares quantization in PCM. IEEE transactions on information theory, 28(2), 129-137.  
[3] Hotelling, H. (1936). Relations between two sets of variates. Biometrika 28, 321–77.


```{toctree}
:hidden:
:titlesonly:


why-embed-networks
random-walk-diffusion-methods
graph-neural-networks
multigraph-representation-learning
joint-representation-learning
```
