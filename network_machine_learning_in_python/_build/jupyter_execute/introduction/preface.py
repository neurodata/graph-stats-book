#!/usr/bin/env python
# coding: utf-8

# # Preface

# ## Network Machine Learning and You

# This book is about networks, and how you can use tools from machine learning to understand and explain them more deeply. Why is this an interesting thing to learn about, and why should you care?
# 
# Well, at some level, every aspect of reality seems to be made of interconnected parts. Atoms and molecules are connected to each other with chemical bonds. Your neurons connect to each other through synapses, and the different parts of your brain connect to each other through groups of neurons interacting with each other. At a larger level, you are interconnected with other humans through social networks, and our economy is a global, interconnected trade network. The Earth's food chain is an ecological network, and larger still, every object with mass in the universe is connected to every other object through a gravitational network.
# 
# So if you can understand networks, you can understand a little something about everything!

# ## Network Machine Learning in Your Projects

# So, naturally you are excited about network machine learning and you would love to join the party!  
# 
# Perhaps you're a researcher and you want to expose shadowy financial networks and corporate fraud? Or create a network framework for measuring teamwork in healthcare? Maybe you're interested in evolutionary releationships between different animals, or maybe you want to model communities of neurons in the brain?
# 
# Or maybe you're a data scientist and your company has tons of data (user logs, financial data, production data, machine sensor data, hotline stats, HR reports, etc.), and more than likely you could
# view the data as a network and unearth some hidden gems of knowledge if you just knew where to look? For example:  
# 
# - Explore purchasing networks and isolate the most active customers  
# - Explore patterns of collaboration in your company's network of employees
# - Detect which transactions are likely to be fraudulent  
# - Isolate groups in your company which are overperforming or underperforming
# - Model the transportation chain necessary to produce and disseminate your product
# - And more  
# 
# Whatever the reason, you have decided to learn about networks and implement their analysis in your projects. Great idea!

# ## Objective and Approach

# This book assumes you know next to nothing about how networks can be viewed as a statistical object. Its goal is to give you the concepts, the intuitions, and the tools you need to actually implement programs capable of learning from network data.
# 
# The book is intended to give you the best introduction you can possibly get to explore and exploit network data. You might be a graduate student, doing research on biochemical networks or trade networks in ancient Mesopotamia. Or you might be a professional interested in an introduction to the field of network data science, because you think it might be useful for your company. Whoever you are, we think you'll find a lot of things that are useful and interesting in this book!
# 
# We'll cover the fundamentals of network data science, focusing on developing intuition on networks as statistical objects, doing so while paired with relevant Python tutorials. By the end of this book, you will be able to utilize efficient and easy to use tools available for performing analyses on networks. You will also have a whole new range of statistical techniques in your toolbox, such as representations, theory, and algorithms for networks.
# 
# We'll spend this book learning about network algorithms by showing how they're implemented in production-ready Python frameworks:
# - Numpy and Scipy are used for scientific programming. They give you access to array objects, which are the main way we'll represent networks computationally.
# - Scikit-Learn is very easy to use, yet it implements many Machine Learning algorithms efficiently, so it makes a great entry point for downstream analysis of networks.
# - Graspologic is an open-source Python package developed by Microsoft and the NeuroData lab at Johns Hopkins University which gives you utilities and algorithms for doing statistical analyses on network-valued data.
# 
# The book favors a hands-on approach, growing an intuitive understanding of
# networks through concrete working examples and a bit of theory.
# While you can read this book without picking up your laptop, we highly recommend
# you experiment with the code examples available online as Jupyter notebooks at [http://docs.neurodata.io/graph-stats-book/index.html](http://docs.neurodata.io/graph-stats-book/index.html).

# ## Prerequisites

# We assume you have a basic knowledge of mathematics. Because network science uses a lot of linear algebra, requiring a bit of linear algebra knowledge is unfortunately unavoidable. (You should know what an eigenvalue is!) 
# 
# If you care about what's under the hood mathematically, we have certain sections marked as "advanced material" - you should have a reasonable understanding of college-level math, such as calculus, linear algebra, probability, and statistics for these sections.
# 
# You should also probably have some background in programming - we'll mainly be using Python to build and explore our networks. If you don't have too much of a Python or math background, don't worry - we'll link some resources to give you a head start.
# 
# If you've never used Jupyter, don't worry about it. It is a great tool to have in your toolbox and it's easy to learn. We'll also link some resources for you if you are not familiar with Python's scientific libraries, like numpy, scipy, networkx, and scikit-learn.

# ## Roadmap

# This book is organized into three parts. 
# 
# Part I, Foundations, gives you a brief overview of the kinds of things you'll be doing in this book, and shows you how to solve a network data science problem from start to finish. It covers the following topics:
# - What a network is and where you can find networks in the wild
# - All the reasons why you should care about studying networks
# - Examples of ways you could apply network data science to your own projects
# - An overview of the types of problems Network Machine Learning is good at dealing with
# - The main challenges you'd encounter if you explored Network Learning more deeply
# - Exploring a real network data science dataset, to get a broad understanding of what you might be able to learn.
# 
# Part II, Representations, is all about how we can represent networks statistically, and what we can do with those representations. It covers the following topics:
# - Ways you can represent individual networks
# - Ways you can represent groups of networks
# - The various useful properties different types of networks have
# - Types of network representations and why they're useful
# - How to represent networks as a bunch of points in space
# - How to represent multiple networks
# - How to represent networks when you have extra information about your nodes
# 
# Part III, Applications, is about using the representations from Part II to explore and exploit your networks. It covers the following topics:
# - Figuring out if communities in your networks are different from each other
# - Selecting a reasonable model to represent your data
# - Finding nodes, edges, or communities in your networks that are interesting
# - Finding time points which are anomalies in a network which is evolving over time
# - What to do when you have new data after you've already trained a network model
# - How hypothesis testing works on networks
# - Figuring out which nodes are the most similar in a pair of networks

# ## Conventions Used In This Book

# ## Using Code Examples

# ## About the Authors

# **Dr. Joshua Vogelstein** is an Assistant Professor in the Department of Biomedical Engineering at Johns Hopkins University, with joint appointments in Applied Mathematics and Statistics, Computer Science, Electrical and Computer Engineering, Neuroscience, and Biostatistics. His research focuses on the statistics of networks in brain science (connectomes). His lab and collaborators have developed the leading computational algorithms and libraries to perform statistical analysis on networks.
# 
# **Alex Loftus** is a master’s student at Johns Hopkins University in the Department of Biomedical Engineering, with an undergraduate degree in neuroscience. He has worked on implementing network spectral embedding and clustering algorithms in Python, and helped develop an MRI pipeline to produce brain networks from diffusion MRI data.
# 
# **Eric Bridgeford** is a PhD student in the Department of Biostatistics at Johns Hopkins University. Eric’s background includes Computer Science and Biomedical Engineering, and he is an avid contributor of packages to CRAN and PyPi for nonparametric hypothesis testing. Eric studies general approaches for statistical inference in network data, with applications to problems with network estimation in MRI connectomics data, including replicability and batch effects.
# 
# **Dr. Carey E. Priebe** is Professor of Applied Mathematics and Statistics, and a founding member of the Center for Imaging Science (CIS) and the Mathematical Institute for Data Science (MINDS) at Johns Hopkins University. He is a leading researcher in theoretical, methodological, and applied statistics / data science; much of his recent work focuses on spectral network analysis and subsequent statistical inference. Professor Priebe is Senior Member of the IEEE, Elected Member of the International Statistical Institute, Fellow of the Institute of Mathematical Statistics, and Fellow of the American Statistical Association.
# 
# **Dr. Christopher M. White** is Managing Director, Microsoft Research Special Projects. He leads mission-oriented research and software development teams focusing on high risk problems. Prior to joining Microsoft, he was a Fellow at Harvard for network statistics and machine learning. Chris’s work has been featured in media outlets including Popular Science, CBS’s 60 Minutes, CNN, the Wall Street Journal, Rolling Stone Magazine, TEDx, and Google’s Solve for X. Chris was profiled in a cover feature for the Sept/Oct 2016 issue of Popular Science.
# 
# **Weiwei Yang** is a Principal Development Manager at Microsoft Research. Her interests are in resource efficient alt-SGD ML methods inspired by biological learning. The applied research group she leads aims to democratize AI by addressing issues of sustainability, robustness, scalability, and efficiency in ML. Her group has applied ML to address social issues such as countering human trafficking and to energy grid stabilizations.
# 

# ## Acknowledgements

# First of all, big thanks to everybody who has been reading the book as we write and giving feedback. So far, this list includes Dax Pryce, Ross Lawrence, Geoff Loftus, Alexandra McCoy, Olivia Taylor, and Peter Brown.

# ## Finished Sections

# (lots more in progress...)

# 1. Preface: [](preface.ipynb)
# 2. Why Use Statistical Models: [](../representations/ch5/why-use-models.ipynb)
# 3. Single-Network Models: [](../representations/ch5/single-network-models.ipynb)
# 4. Multi-Network Representation Learning: [](../representations/ch6/multigraph-representation-learning.ipynb)
# 5. Joint Representation Learning: [](../representations/ch6/joint-representation-learning.ipynb)
