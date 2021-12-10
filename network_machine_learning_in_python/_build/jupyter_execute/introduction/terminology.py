#!/usr/bin/env python
# coding: utf-8

# # Terminology and Math Refresher
# 
# In this section, we outline some background terminology which will come up repeatedly throughout the book. This section attempts to standardize some background material that we think is useful going in. It is important to realize that many of the concepts discussed below are only crucial for understanding the advanced, starred sections. If you aren't familiar with some (or any!) of the below concepts, we don't think this would detract from your understanding of the broader content.
# 
# 
# ## Vectors, Matrices, and Numerical Spaces
# 
# Throughout this book, we will need some level of familiarity with numerical spaces, and the grammar that we use to describe them. Taking the time to understand this notation will better help you understand many of the concepts in the rest of the book.
# 
# ### Numerical Spaces
# 
# Numerical spaces are everywhere. If you have taken any calculus or algebra courses, you are likely familiar with the natural numbers - these are just your basic one, two, three, and so on. This constitutes the most basic numerical space and is denoted by the symbol $\mathbb N$. Formally, the natural numbers describes the set:
# \begin{align*}
#     \mathbb N &= \{1, 2, 3, ...\}
# \end{align*}
# and continues infinitely (notice that neither negative numbers nor numbers with decimal points appear in $\mathbb N$). On a similar note, we will frequently resort to short hand to describe subsets of the natural numbers. We will use the symbol $\in$ (read, "in") to denote that one quantity is found within a particular set. For example, since $5$ is a natural number, we would say that $5 \in \mathbb N$, which can be thought of as "$5$ is in the set of natural numbers". To describe a subset of the first $5$ natural numbers, we would use the notation $[5]$, which denotes the set:
# \begin{align*}
#     [5] &= \{1,2,3,4,5\}
# \end{align*}
# In the more general case where we have some variable $n$ where $n \in \mathbb N$ (again, $n$ is some arbitrary natural number), then:
# \begin{align*}
#     [n] &= \{1,2,...,n\}
# \end{align*}
# 
# The next most basic numerical space is known as the integers, which is just the natural numbers combined with the negative numbers and zero. Specifically:
# 
# \begin{align*}
#     \mathbb Z &= \{..., -2, -1, 0, 1, 2, ...\}
# \end{align*}
# From $-\infty$ up to $+\infty$. 
# 
# There are many more numerical spaces, but in this book we'll focus on one in particular: real numbers, denoted $\mathbb R$. The real numbers can be thought of as all the numbers that can be represented by a finite or infinite number of decimal places in between (and including) the integers. We won't go into too many details; if you want more details on the real numbers, a good place to start would be coursework in **real analysis**. Particularly, the real numbers include any natural number, and integer, any decimal, or any irrational number (such as $\pi$ or $\sqrt{2}$). The main thing that is interesting about the real numbers that we will *indirectly* use throughout the book is that if we have any two real numbers $x$ and $y$ (remember, this would be written $x, y \in \mathbb R$), then the products, ratios, or sums of them are also real numbers:
# \begin{align*}
#     x \cdot y, \frac{x}{y}, x + y \in \mathbb R
# \end{align*}
# 
# Throughout the book, we will build upon some of these numerical spaces and introduce several new ones along the way that are interesing for network machine learning. We will do this by attempting to relate them back to the basic numerical spaces we have introduced here. 
# 
# ### One-Dimensional Quantities
# 
# We will frequently see the term "dimensional" come up in this book, and we will attempt to give some insight into what this means here. If we were to say that $x \in \mathbb R$, we know from the above description that this means that $x$ is a real number, and is therefore "in" the set of real numbers. A one-dimensional quantity is a quantity which is described by a single element from one numerical space. In this instance, $x$ is described by one real number, and is therefore one-dimensional. We will use a lowercase letter (for instance, $x, a, b, \alpha, \beta$; the letters may be Roman or Greek) to denote that a quantity is one-dimensional.
# 
# 
# ### Vectors
# 
# Building off the concept of one-dimensional variables, what if we had some variable that existed in two dimensions? For instance, consider the following:
# \begin{align*}
#     \vec x = \begin{bmatrix}1.5 \\ 2\end{bmatrix}
# \end{align*}
# As we can see here, $\vec x$ is now described by two real numbers (namely, $1.5$ and $2$). This means that $\vec x$ is now a two-dimensional quantity, since we have two separate values needed to describe $\vec x$. In this case, $\vec x$ no longer is "in" the real numbers, it is instead in the two-dimensional real vectors, or $\mathbb R^2$. Here, $\vec x$ is called a **vector**, and each of its dimensions are defined using the notation $x_1 = 1.5$ and $x_2 = 2$. The subscript $x_j$ just means the $j^{th}$ element of $\vec x$, which is numbered by counting downwards from the first row ($j = 1$) to however many rows $\vec x$ has in total. Since $\vec x$ is two-dimensional, we would say that $j \in [2]$, which means $j$ can be either $1$ or $2$. In general, we will assume that all vectors are **column vectors** unless otherwise stated, which means that $\vec x$ will be assumed to be vertically aligned. This will not make much of a conceptual difference, but it will play a role when we define operations between vectors and matrices later on. On the other hand, a **row vector** will typically be denoted by using the **transpose** symbol, which we will learn about later on in the section on operators. Unlike a column vector, a row vector is aligned horizontally. For example, a row vector with entries identical to $\vec x$ will be denoted:
# \begin{align*}
#     \vec x^\top = \begin{bmatrix}1.5 & 2\end{bmatrix}
# \end{align*}
# 
# In the general case, for any set $\mathcal S$, we would say that $\vec s \in \mathcal S^d$ if (think through this notation!) for any $j \in [d]$, $s_j \in \mathcal S$. The key aspects are that the symbol for the vector will be a lower case letter (in this example, $s$) like the one-dimensional quantity, but will add the $\vec{}$ symbol to denote that it is a vector with more than one dimension. The quantity $d$ that you see in the superscript is referred to as the dimensionality. In this example, we would say that $\vec s$ is a $d$-dimensional $\mathcal S$-vector. 
# 
# ### Matrices
# 
# Matrices come up a lot in network science because we often represent networks as matrices: the adjacency matrix, for instance, is a way to represent a network in terms of its edge connections. Because networks can be represented as matrices, we'll sometimes just talk about matrices directly.
# 
# We will see a variety of different types of matrices throughout this book, so let's start with a simple example. Consider the following marix:
# \begin{align*}
#     X = \begin{bmatrix}
#     1.5 & 1.7 \\
#     2 & 1.8
#     \end{bmatrix}
# \end{align*}
# Here, we can see that $X$ is described by four real numbers, with a particular arrangement. This time, we say that $X$ is an element of the set of all possible $2 \times 2$ (2 rows, and 2 columns) matrices with real entries. In symbols, we would describe this as $\mathbb R^{2 \times 2}$, where $\mathbb R$ says that the elements of the matrix are real numbers, and $2 \times 2$ means that the matrix has two rows and two columns. We can describe the entries of a matrix using indexing, very similar to what we did for vectors. In matrices, the rows and columns matter. In this case, the rows go from left to right horizontally, and the columns go from top to bottom vertically. The rows will be numbered from the top of the matrix to the bottom, and the columns will be numbered from the left-most column to the right-most column. For instance, the first row of the matrix $X$ is the row-vector $\begin{bmatrix}1.5 & 1.7\end{bmatrix}$, and the second column of the matrix $X$ is the column-vector $\begin{bmatrix}1.7 \\ 1.8\end{bmatrix}$. This subscripts $x_{ij}$ means the entry of the matrix $X$ in the $i^{th}$ row aand the $j^{th}$ column. In this instance, we would describe that $x_{11} = 1.5$, $x_{12} = 1.7$, $x_{21}=2$, and $x_{22} = 1.8$.
# 
# In the general case, for a set $\mathcal S$, we would say that $S \in \mathcal S^{r \times c}$ if (think this through!) for any $i \in [r]$ and any $j \in [c]$, $s_{ij} \in \mathcal S$. Like before, the key aspects are that the symbol for a matrix will be a capital letter (in this example, $S$) to denote that it is a matrix, and its entries $s_{ij}$ will be denoted using a lowercase letter. The quantity $r$ is known as the row count and the quantity $c$ is known as the column count of the matrix $S$. In this example, we would say that $S$ is a $\mathcal S$-matrix with $r$ rows and $c$ columns.
# 
# Another thing we will see arise periodically is that vectors can be denoted as matrices with a single column. For example, in our example above in the vector section, we might equivalently write that $\vec s \in \mathcal S^{d \times 1}$. The "1" for the columns just denotes that $\vec s$ is a column vector with $d$ rows in total. This will be useful when we define functions for matrices, and use the same notation for functions on vectors.
# 
# ## Useful Functions
# 
# Throughout the book, we will deal with many types of functions which take mathematical objects (potentially multiple) that exist in one numerical space and produce a mathematical object (potentially in a different) numerical space. You are probably familiar with several of these, such as the addition or multiplication operators on one-dimensional quantities. We will touch on some of the more fancy ones that we will see arise throughout the book.
# 
# The **sum**, denoted by a fancy capital epsilon $\sum$, denotes that we are summing a bunch of items which can be easily indexed. For instance, consider if we have a vector $\vec x \in \mathbb R^d$, so $\vec x$ is a $d$-dimensional vector. If we wanted to take the sum of all of the elements of $\vec x$, we would write:
# \begin{align*}
#     \sum_{i = 1}^d x_i = x_1 + x_2 + ... + x_d
# \end{align*}
# The *summand* of the sum, the $x_i$s next to the $\sum$ symbol, are the terms that will be summed up. Further, note that the $\sum$ symbol also indicates the indices of $\vec x$ that will be summed. Note that on the bottom, we see that the sum says from $i = 1$ and above it says $d$. This means that we sum all the elements of $x_i$ starting from below at $1$ and going up until $d$. We could say the exact same thing using our shorthand for this set, which we described in the section on natural numbers, $[d]$:
# \begin{align*}
# \sum_{i \in [d]} x_i = \sum_{i = 1}^d x_i = x_1 + x_2 + ... + x_d
# \end{align*}
# We could similarly define **any** indexing set, such as $\mathcal I = \{1,3\}$, and write:
# \begin{align*}
#     \sum_{i \in \mathcal I} x_i = x_1 + x_3
# \end{align*}
# The key is that the notation above or below the summand just tells us which elements we are applying the sum over. For instance, if $\vec x$ was a $3$-dimensional vector:
# \begin{align*}
#    \vec x = \begin{bmatrix}
#       1.7 \\ 1.8 \\ 2
#    \end{bmatrix}
# \end{align*}
# We would have that:
# \begin{align*}
#    \sum_{i = 1}^3 x_i = 5.5
# \end{align*}
# if we were to use $\mathcal I = \{1,3\}$, then:
# \begin{align*}
#     \sum_{i \in \mathcal I}x_i = 3.7
# \end{align*}
# 
# the **product**, denoted by a capital pi $\prod$, behaves extremely similarly to the sum, except insted of applying sums, it applies multiplication. For instance, if we instead wanted to multiply all the elements of $\vec x$, we would write:
# \begin{align*}
#     \prod_{i = 1}^d x_i = x_1 \times x_2 \times ... \times x_d
# \end{align*}
# Where $\times$ is just multiplication like you are probably used to. Again, we have the exact same indexing conventions, where:
# \begin{align*}
#     \prod_{i \in [d]} x_i=
#     \prod_{i = 1}^d x_i = x_1 \times x_2 \times ... \times x_d
# \end{align*}
# We can again just use indexing sets, too:
# \begin{align*}
#     \prod_{i \in \mathcal I}x_i = x_1 \times x_3
# \end{align*}
# With $\vec x$ defined as above in the sum example, we would have that:
# \begin{align*}
#    \prod_{i = 1}^3 x_i = 6.12
# \end{align*}
# if we were to use $\mathcal I = \{1,3\}$, then:
# \begin{align*}
#     \prod_{i \in \mathcal I}x_i = 3.4
# \end{align*}
# 
# The **Euclidean inner product**, or the *inner product* we will refer to in our book, is obtained by multiplying two vectors element-wise, and summing the result. Suppose we have two vectors $\vec x$ and $\vec y$, which are each $d$-dimensional real vectors (both $x$ and $y$ must have the same number of elements). The inner product is the quantity:
# \begin{align*}
#     \langle \vec x, \vec y\rangle &= \sum_{i = 1}^d x_i y_i
# \end{align*}
# as we will see in a second, in matrix notation, this is exactly equivalent to writing:
# \begin{align*}
#     \langle \vec x, \vec y\rangle &= \vec x^T \vec y
# \end{align*}
# 
# **Matrix multiplication**, denoted by a circle $\cdot$ (or in most cases, just two matrices side by side, with no separation), is an operation which takes a matrix which has $r$ rows and $c$ columns and another matrix which has $c$ rows and $l$ columns, and produces a matrix with $r$ rows and $l$ columns. Suppose we have a matrix $A \in \mathbb R^{r \times c}$, and $B \in \mathbb R^{c \times l}$. Here, $r$, $c$, and $l$ could be *any* natural numbers. A matrix multiplication produces a matrix $D \in \mathbb R^{r \times l}$, where:
# \begin{align*}
#     d_{ij} = \sum_{k = 1}^c a_{ik}b_{kj}
# \end{align*}
# What does this mean intuitively? Well, let's think about it. Let's imagine that the *rows* of $A$ are indexed from $1$ to $r$, like this:
# \begin{align*}
#     A &= \begin{bmatrix}
#         \vec a_1^T \\
#         \vec a_2^T \\
#         \vdots \\
#         \vec a_r^T
#     \end{bmatrix}
# \end{align*}
# Note that the vectors $\vec a_i$ are transposed when oriented in the matrix $A$, because they are each $c$-dimensional vectors (and by convention in our book, all vectors will be *column* vecors. So to comprise the rows of $A$, they must be "flipped"). Similarly, let's imagine that the columns of $B$ are indexed from $1$ to $l$, like this:
# \begin{align*}
#     B &= \begin{bmatrix}
#         \vec b_1 & \vec b_2 & ... & \vec b_l
#     \end{bmatrix}
# \end{align*}
# So what is the marix $D$? Note that each entry, $d_{ij} = \langle \vec a_i, \vec b_j\rangle = \vec a_i^T \vec b_j$. So the matrix $D$ is the matrix whose entries are the *inner products of the rows of $A$ with the columns of $B$*. In a diagram, $D$ is like this:
# \begin{align*}
#     D &= \begin{bmatrix}
#         \vec a_1^T\vec b_1 & ... & \vec a_1^T \vec b_l \\
#         \vdots & \ddots & \vdots \\
#         \vec a_r^T \vec b_1 & ... & \vec a_r^T \vec b_l
#     \end{bmatrix}
# \end{align*}
# As a matter of notation, we might often have the case where we want to discuss or interpret a single element which is a product of two matrices. For instance, suppose we care about the entry $(i, j)$ of $AB$. We might also describe the resulting quantity $d_{ij}$ using the notation $(AB)_{ij}$. The reason we adopt this notation is that we want to emphasize that the matrix multiplication operation is performed first (it is in *parentheses*), and then we look at the $(i,j)$ entry of the resulting matrix. 
# 
# The **Euclidean distance** is the most common distance between vectors we will see in this book. The Euclidean distance effectively tells us how far apart two points in $d$-dimensional space are. Given $\vec x, \vec y \in \mathbb R^d$ ($\vec x$ and $\vec y$ are $d$-dimensional real vectors), the Euclidean distance is the quantity:
# \begin{align*}
#     \delta(\vec x, \vec y) &= \langle \vec x - \vec y, \vec x - \vec y\rangle = \sum_{i = 1}^d (x_i - y_i)^2
# \end{align*}
# In particular, if we check the distance between a vector and the origin (the **zero-vector**, denoted $0_d$, which is a $d$-dimensional vector where all entries are $0$), we end up with a very useful quantity, called the squared Euclidean norm. We will use a special notation for the Euclidean norm, which is:
# \begin{align*}
#     ||\vec x||_2^2 &= \delta(\vec x, 0_d) = \sum_{i = 1}^dx_i^2
# \end{align*}
# The subscript $_2$ just means that this is the "2"-norm, which is a concept outside of the scope of this book. The superscript $^2$ means that this is the squared Euclidean norm. Therefore, the Euclidean norm itself is:
# \begin{align*}
# ||\vec x||_2 &= \sqrt{\delta(\vec x, 0_d)} = \sqrt{\sum_{i = 1}^d x_i^2}
# \end{align*}
# What does this mean interpretation wise? The "square" operation basically means, if there are dimensions of $\vec x$ that are big, the norm will end up being big. If the dimensions of $\vec x$ are small, they will not contribute very much to the norm.
# 
# Based on the equation we saw above for the Euclidean distance, we could also understand the Euclidean distance to be the squared Euclidean norm of the vector which is the difference between $\vec x$ and $\vec y$. Using this convention:
# \begin{align*}
#     \delta(\vec x, \vec y) &= ||\vec x - \vec y||_2^2
# \end{align*}
# In this sense, we can see that the Euclidean distance and the Euclidean norms are attributing a concept of "length" and "how far" a vector is from another (whether that is the origin or an arbitrary real vector). Next, we will see a related concept for matrices. The **squared Frobenius norm** is the quantity, given a matrix $A \in \mathbb R^{r \times c}$:
# \begin{align*}
#     ||A||_F^2&= \sum_{i = 1}^r \sum_{i = 1}^c a_{ij}^2
# \end{align*}
# Note that this is very similar to the squared Euclidean norm of a vector, except it is applied to both the rows *and* the columns of $A$. Again, we have a similar interpretation to the Euclidean norm. If an entry of $A$ is big, it will contribute much to the Frobenius norm due to the squared $a_{ij}$ term. If an entry is smaller, it will not contribute as much. The Frobenius norm itself is just the square root of this:
# \begin{align*}
#     ||A||_F &= \sqrt{\sum_{i = 1}^r \sum_{i = 1}^c a_{ij}^2}
# \end{align*}
# 
# ## Probability
# 
# Throughout this book, we will be very concerned with probabilities and probability distributions. For this reason, we will introduce some basic notation that we will be concerned with. In probability analyses, we are concerned with describing things that occur in the real world with some level of uncertainty. We capture this uncertainty using probability, which in essence, describes how likely (or unlikely) a particular outcome is compared to all of the possible outcomes that could be realized. In general, we will call the most basic objects which occur with some uncertainty **random variables**, which is a variable whose values that we get to see in the real world (the *realizations* of the random variable) depend on some random phenomenon. We will denote a random variable using a similar notation to a one-dimensional variable, with the exception that we will *bold face* the variable to make clear that it is random. For instance, for a one-dimensional random variable, we will use notation like $\mathbf x$.
# 
# Like before, we can also have random vectors and random matrices. Like for the random variable, we will denote these with bold faces too. A random vector will be denoted using a bold faced variable with the vector symbol; for example, $\vec{\mathbf x}$. Likewise, a random matrix will be denoted using a bold faced upper case letter; for example, $\mathbf X$. Similar to how we indexed vectors and matrices, the index positions of random vectors and random matrices are random variables, too. That is, $\vec{\mathbf x}$ is a $d$-dimensional random vector whose entries are the random variables $\mathbf x_i$ for all $i$ from $1$ to $d$:
# \begin{align*}
#     \vec{\mathbf x} &= \begin{bmatrix}
#         \mathbf x_1 \\
#         \vdots \\
#         \mathbf x_d
#     \end{bmatrix}
# \end{align*}
# And $\mathbf X$ is a $(r \times c)$ random matrix whose entries are the random varaiables $\mathbf x_{ij}$ for all $i$ from $1$ to $r$ and $j$ from $1$ to $c$:
# \begin{align*}
#     \mathbf X &= \begin{bmatrix}
#         \mathbf x_{11} & ... & \mathbf x_{1c} \\
#         \vdots & \ddots & \vdots \\
#         \mathbf x_{r1} & ... & \mathbf x_{rc}
#     \end{bmatrix}
# \end{align*}
# 
# A probability distribution, denoted by $\mathbb P$, is a function which gives the probability of a particular value being attained by a random quantity. To state this another way, the probability distribution is concerned with fixing probabilities to realizations of random quantities. to make this a little more concrete, we will give an example with the simplest possible probability distribution, the Bernoulli distribution, denoted $Bernoulli(p)$. For the sake of this example, we will say that $\mathbf x$ is a random variable which is $Bernoulli(p)$ distributed, which we denote by $\mathbf x \sim Bernoulli(p)$. The Bernoulli distribution describes that the probability of the random variable $\mathbf x$ taking a realization of $1$ is $p$, whereas the probability of the random variable $\mathbf x$ taking a realization of $0$ is $1 - p$. Using the probability distribution, we would say that:
# \begin{align*}
#     \mathbb P(\mathbf x = 0) &= 1 - p \\
#     \mathbb P(\mathbf x = 1) &= p
# \end{align*}
# 
# ## Advanced Probability*
# 
# the probability distribution for a random vector or a random matrix is described very similarly. The caveat is that with a random vector/matrix, we affix a probability of *every element* of the random vector/matrix equaling the realized vector/matrix. For instance, if $\vec{\mathbf x}$ is a random vector taking realizations which are $d$-dimensional vectors, and $\vec x$ is one such $d$-dimensional vector, then:
# \begin{align*}
#     \mathbb P(\vec{\mathbf x} = \vec x) = \mathbb P(\mathbf x_1 = x_1, ..., \mathbf x_d = x_d)
# \end{align*}
# and likewise, if $\mathbf X$ is a random matrix taking realizations which are $r \times c$ matrices, and $X$ is one such $r \times c$ matrix, then:
# \begin{align*}
#     \mathbb P(\mathbf X = X) &= \mathbb P(\mathbf x_{11} = x_{11}, ..., \mathbf x_{rc} = x_{rc}) \\
#     &= \mathbb P(\mathbf x_{ij} = x_{ij} \text{ for any }i\text{ and }j)
# \end{align*}
# 
# A probability concept we will see arise frequently in the advanced sections of the book is one called independence. A pair of random variables are independent if for any $x$ which is a possible realization of $\mathbf x$ and $y$ is a possible realization of $\mathbf y$, then:
# \begin{align*}
#     \mathbb P(\mathbf x = x, \mathbf y = y) = \mathbb P(\mathbf x = x) \mathbb P(\mathbf y = y)
# \end{align*}
# A related concept that will be very important in our study of random matrices is the idea of mutual independence. If we have a set of $n$ random variables $\mathbf x_i$ for all $i = 1,..., n$, this set of random variables is said to be mutually independent if for any $x_1$ which is a possible realization of $\mathbf x_1$, any $x_2$ which is a possible realization of $\mathbf x_2$, and so on up to $\mathbf x_n$, then:
# \begin{align*}
#     \mathbb P(\mathbf x_1 = x_1, ..., \mathbf x_n = x_n) = \prod_{i = 1}^n \mathbb P(\mathbf x_i = x_i)
# \end{align*}
# The ways in which this is useful will become more obvious through some of the advanced material of later chapters.
# 
# Another important concept we will see arise in some of the advanced material is the idea of conditional distributions. Given $x$ which is a possible realization of $\mathbf x$ and $y$ is a possible realization of $\mathbf y$, then the conditional distribution of $\mathbf x$ on $\mathbf y$ is the quantity:
# \begin{align*}
# \mathbb P(\mathbf x = x | \mathbf y = y) &= \frac{\mathbb P(\mathbf x = x, \mathbf y = y)}{\mathbb P(\mathbf y = y)}
# \end{align*}
# While outside the scope of this book, it can be shown that this is a proper probability distribution function, but we mainly are concerned with the fact that this is simply a useful notation for an intuitive idea. What this allows us to capture is the idea of attributing a probability for a random variable $\mathbf x$ obtaining the value $x$, given that we already know that $\mathbf y$ obtains the value $y$. A related concept, Baye's Rule, uses a simple consequence of this theorem. Note that we could flip the probability statement above, and would obtain that:
# \begin{align*}
# \mathbb P(\mathbf y = y | \mathbf x = x) &= \frac{\mathbb P(\mathbf x = x, \mathbf y = y)}{\mathbb P(\mathbf x = x)}
# \end{align*}
# a simple rearrangement of terms by multiplying both sides by $\mathbb P(\mathbf x = x)$ gives us that:
# \begin{align*}
# \mathbb P(\mathbf x, \mathbf y)&= 
# \mathbb P(\mathbf y = y | \mathbf x = x)\mathbb P(\mathbf x = x)
# \end{align*}
# Substituting this in to our first definition for a conditional distribution of $\mathbf x$ on $\mathbf y$ gives:
# \begin{align*}
# \mathbb P(\mathbf x = x | \mathbf y = y) &= \frac{\mathbb P(\mathbf y = y | \mathbf x = x)\mathbb P(\mathbf x = x)}{\mathbb P(\mathbf y = y)}
# \end{align*}
# which is Baye's Rule.

# In[ ]:




