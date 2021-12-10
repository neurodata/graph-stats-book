#!/usr/bin/env python
# coding: utf-8

# In[1]:


## What The Heck Is The Latent Position Matrix, Anyway?


# In[ ]:





# ## When Should You Use ASE vs LSE?

# You'll note that both Adjacency Spectral Embedding - which embeds using only the adjacency matrix - and Laplacian Spectral Embedding - which embeds using the Laplacian - are both reasonable options to embed your network. When should you use one compared to the other?

# In[2]:


# Generate a network from an SBM
B = np.array([[0.02, 0.044, .002, .009], 
              [0.044, 0.115, .010, .042],
              [.002, .010, .020, .045],
              [.009, .042, .045, .117]])
n = [100, 100, 100, 100]
A, labels = sbm(n=n, p=B, return_labels=True)

# Instantiate an ASE model and find the embedding
ase = ASE(n_components=2)
embedding_ase = ase.fit_transform(A)

# LSE
lse = LSE(n_components=2)
embedding_lse = lse.fit_transform(A)

# plot
from graphbook_code import draw_layout_plot

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plot_latents(embedding_ase, labels=labels, ax=axs[0],
             title="Adjacency Spectral Embedding");
plot_latents(embedding_lse, labels=labels, ax=axs[1],
             title="Laplacian Spectral Embedding");

