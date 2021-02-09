---
layout:	"post"
categories:	"machine_learning"
title:	"Separating mixed signals with Independent Component Analysis"
date:	2019-04-29
author:	Carsten Klein
---

* * *

![_config.yml]({{ site.baseurl }}/images/ICA/1f6zxmgbeke5RJCCiM7pdBw.jpeg)Image
modified from [garageband](https://pixabay.com/users/garageband-4200899/)

The world around is a dynamic mixture of signals from various sources. Just
like the colors in the above picture blend into one another, giving rise to
new shades and tones, everything we perceive is a fusion of simpler
components. Most of the time we are not even aware that the world around us is
such a chaotic intermix of independent processes. Only in situations where
different stimuli, that do not mix well, compete for our attention we realize
this mess. A typical example is the scenario at a cocktail party where one is
listening to the voice of another person while filtering out the voices of all
the other guests. Depending on the loudness in the room this can either be a
simple or a hard task but somehow our brains are capable of separating the
signal from the noise. While it is not understood how our brains do this
separation there are several computational techniques out there that aim at
splitting a signal into its fundamental components. One of these methods is
termed **I** ndependent **C** omponent **A** nalysis ( _ICA_ ) and here we
will have a closer look on how this algorithm works and how to write it down
in Python code. If you are more interested in the code than in the explanation
you can also directly check out [the Jupyter Notebook for this post on
Github](https://github.com/akcarsten/Independent_Component_Analysis).

### What is Independent Component Analysis?

Lets stay with the example of the cocktail party for now. Imaging there are
two people talking, you can hear both of them but one is closer to you than
the other. The sound waves of both sources will mix and reach your ears as a
combined signal. Your brain will un-mix both sources and you will perceive the
voices of both guests separately with the one standing closer to you as the
louder one. Now lets describe this in a more abstract and simplified way. Each
source is a sine wave with a constant frequency. Both sources mix depending on
where you stand. This means the source closer to you will be more dominant in
the mixed signal than the one more far away. We can write this down as follows
in vector-matrix notation:

![_config.yml]({{ site.baseurl }}/images/ICA/1l0DEYdLGe0Gc12qn2I-w3w.png)

Where _x_ is the observed signal, _s_ are the source signals and _A_ is the
mixing matrix. In other words our model assumes that the signals _x_ are
generated through a linear combination of the source signals. In Python code
our example will look like this:



    >> import numpy as np


    >>> # Number of samples   
    >>> ns = np.linspace(0, 200, 1000)


    >>> # Sources with (1) sine wave, (2) saw tooth and (3) random noise  
    >>> S = np.array([np.sin(ns * 1),   
                      signal.sawtooth(ns * 1.9),  
                      np.random.random(len(ns))]).T


    >>> # Quadratic mixing matrix  
    >>> A = np.array([[0.5, 1, 0.2],  
                      [1, 0.5, 0.4],  
                      [0.5, 0.8, 1]])


    >>> # Mixed signal matrix  
    >>> X = S.dot(A).T

As can be seen from the plots in _Figure 1_ below the code generates one sine
wave signal, one saw tooth signal and some random noise. These three signals
are our independent sources. In the plot below we can also see the three
linear combinations of the source signals. Further we see that the first mixed
signal is dominated by the saw tooth component, the second mixed signal is
influence more by the sine wave component and the last mixed signal is
dominated by the noise component.

![_config.yml]({{ site.baseurl }}/images/ICA/1ORG-sPyzod3wEv1lt_RS-Q.png)
![_config.yml]({{ site.baseurl }}/images/ICA/1UmC-AzkMb3U58B9-2t8xYg.png)Figure
1: Source signals (upper plots) and linear combinations of the source signals
(lower plots).

Now, according to our model we can retrieve the source signals again from the
mixed signals by multiplying _x_ with the inverse of _A_ :

![_config.yml]({{ site.baseurl }}/images/ICA/19C7yXxG4oYCV8fM1hO6H4w.png)
![_config.yml]({{ site.baseurl }}/images/ICA/1mZDPc-E0swUtzVeShEXxyg.png)

This means in order to find the source signals we need to calculate _W_. So
the task for the rest of this post will be to find _W_ and retrieve the three
independent source signals from the three mixed signals.

### Preconditions for the ICA to work

Now, before we continue we need to think a little more about what properties
our source signals need to have so that the ICA successfully estimates _W_.
The **first precondition** for the algorithm to work is that the mixed signals
are a linear combination of any number of source signals. The **second
precondition** is that the source signals are independent. So what does
independence mean? Two signals are independent if the information in signal
_s1_ does not give any information about signal _s2_. This implies that they
are not correlated, which means that their covariance is 0. However, one has
to be careful here as uncorrelatedness does not automatically mean
independence. The **third precondition** is that the independent components
are non-Gaussian. Why is that? The joint density distribution of two
independent non-Gaussian signals will be uniform on a square; see upper left
plot in _Figure 2_ below. Mixing these two signals with an orthogonal matrix
will result in two signals that are now not independent anymore and have a
uniform distribution on a parallelogram; see lower left plot in _Figure 2_.
Which means that if we are at the minimum or maximum value of one of our mixed
signals we know the value of the other signal. Therefore they are not
independent anymore. Doing the same with two Gaussian signals will result in
something else (see right panel of _Figure 2_ ). The joint distribution of the
source signals is completely symmetric and so is the joint distribution of the
mixed signals. Therefore it does not contain any information about the mixing
matrix, the inverse of which we want to calculate. It follows that in this
case the ICA algorithm will fail.

![_config.yml]({{ site.baseurl }}/images/ICA/1SUG1rrf1M_AVSYLWHFnSGA.png)Figure
2: Gaussian and non-Gaussian sources and their mixtures

So in summary for the ICA algorithm to work the following preconditions need
to be met: Our sources are a ( **1** ) lineare mixture of ( **2** )
independent, ( **3** ) non-Gaussian signals.

So lets quickly check if our test signals from above meet these preconditions.
In the left plot below we see the sine wave signal plottet against the saw
tooth signal while the color of each dot represents the noise component. The
signals are distributed on a square as expected for non-Gaussian random
variables. Likewise the mixed signals form a parallelogram in the right plot
of Figure 3 which shows that the mixed signals are not independent anymore.

![_config.yml]({{ site.baseurl }}/images/ICA/16IC6F-nZJc_bOtvu2QZMaA.png)Figure 3: Scatter plots of source and mixed
signals

### Pre-processing steps

Now taking the mixed signals and feeding them directly into the ICA is not a
good idea. To get an optimal estimate of the independent components it is
advisable to do some pre-processing of the data. In the following the two most
important pre-processing techniques are explained in more detail.

#### Centering

The first pre-processing step we will discuss here is _centering_. This is a
simple subtraction of the mean from our input _X_. As a result the centered
mixed signals will have zero mean which implies that also our source signals
_s_ are of zero mean. This simplifies the ICA calculation and the mean can
later be added back. The _centering_ function in Python looks as follows.



    >>> def center(x):  
    >>>     return x - np.mean(x, axis=1, keepdims=True)

#### Whitening

The second pre-processing step that we need is _whitening_ of our signals _X_.
The goal here is to linearly transform _X_ so that potential correlations
between the signals are removed and their variances equal unity. As a result
the covariance matrix of the whitened signals will be equal to the identity
matrix:

![_config.yml]({{ site.baseurl }}/images/ICA/1fUmzOxWw-KUIkVtyfvWA3A.png)

Where _I_ is the identity matrix. Since we also need to calculate the
covariance during the whitening procedure we will write a small Python
function for it.



    >>> def covariance(x):  
    >>>     mean = np.mean(x, axis=1, keepdims=True)  
    >>>     n = np.shape(x)[1] - 1  
    >>>     m = x - mean  
    >>> return (m.dot(m.T))/n

The code for the whitening step is shown below. It is based on the Singular
Value Decomposition (SVD) of the covariance matrix of _X_. If you are
interested in the details of this procedure I recommend [this
article](https://machinelearningmastery.com/singular-value-decomposition-for-
machine-learning/).



    >>> def whiten(x):  
    >>>     # Calculate the covariance matrix  
    >>>     coVarM = covariance(X)


    >>>     # Singular value decoposition  
    >>>     U, S, V = np.linalg.svd(coVarM)  

    >>>     # Calculate diagonal matrix of eigenvalues  
    >>>     d = np.diag(1.0 / np.sqrt(S))   

    >>>     # Calculate whitening matrix  
    >>>     whiteM = np.dot(U, np.dot(d, U.T))  

    >>>     # Project onto whitening matrix  
    >>>     Xw = np.dot(whiteM, X)   

    >>>     return Xw, whiteM

### Implementation of the FastICA Algorithm

OK, now that we have our pre-processing functions in place we can finally
start implementing the ICA algorithm. There are several ways of implementing
the ICA based on the contrast function that measures independence. Here we
will use an _approximation of_ _negentropy_ in an ICA version called FastICA.

So how does it work? As discussed above one precondition for ICA to work is
that our source signals are non-Gaussian. An interesting thing about two
independent, non-Gaussian signals is that their sum is more Gaussian than any
of the source signals. Therefore we need to optimize _W_ in a way that the
resulting signals of _Wx_ are as non-Gaussian as possible. In order to do so
we need a measure of gaussianity. The simplest measure would be _kurtosis_ ,
which is the fourth moment of the data and measures the "tailedness" of a
distribution. A normal distribution has a value of 3, a uniform distribution
like the one we used in _Figure 2_ has a kurtosis < 3\. The implementation in
Python is straight forward as can be seen from the code below which also
calculates the other moments of the data. The first moment is the mean, the
second is the variance, the third is the skewness and the fourth is the
kurtosis. Here 3 is subtracted from the fourth moment so that a normal
distribution has a kurtosis of 0.



    >>> def kurtosis(x):  
    >>>     n = np.shape(x)[0]  
    >>>     mean = np.sum((x**1)/n) # Calculate the mean  
    >>>     var = np.sum((x-mean)**2)/n # Calculate the variance  
    >>>     skew = np.sum((x-mean)**3)/n # Calculate the skewness  
    >>>     kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis  
    >>>     kurt = kurt/(var**2)-3


    >>> return kurt, skew, var, mean

For our implementation of ICA however we will not use kurtosis as a contrast
function but we can use it later to check our results. Instead we will use the
following contrast function _g(u)_ and its first derivative _g '(u)_:

![_config.yml]({{ site.baseurl }}/images/ICA/17FrCPS9rj7Ik3XboaTlfLA.png)

The FastICA algorithm uses the two above functions in the following way in a
__[fixed-point iteration
scheme](https://homepage.math.uiowa.edu/~whan/072.d/S3-4.pdf):

![_config.yml]({{ site.baseurl }}/images/ICA/1oAoWoQBk5sicc83iXLtqqw.png)

So according to the above what we have to do is to take a random guess for the
weights of each component. The dot product of the random weights and the mixed
signals is passed into the two functions _g_ and _g '_. We then subtract the
result of _g '_ from _g_ and calculate the mean. The result is our new weights
vector. Next we could directly divide the new weights vector by its norm and
repeat the above until the weights do not change anymore. There would be
nothing wrong with that. However the problem we are facing here is that in the
iteration for the second component we might identify the same component as in
the first iteration. To solve this problem we have to decorrelate the new
weights from the previously identified weights. This is what is happening in
the step between updating the weights and dividing by their norm. In Python
the implementation then looks as follows:



    >>> def fastIca(signals,  alpha = 1, thresh=1e-8, iterations=5000):  
    >>>     m, n = signals.shape


    >>>     # Initialize random weights  
    >>>     W = np.random.rand(m, m)


    >>>     for c in range(m):  
    >>>             w = W[c, :].copy().reshape(m, 1)  
    >>>             w = w/ np.sqrt((w ** 2).sum())


    >>>             i = 0  
    >>>             lim = 100  
    >>>             while ((lim > thresh) & (i < iterations)):


    >>>                 # Dot product of weight and signal  
    >>>                 ws = np.dot(w.T, signals)


    >>>                 # Pass w*s into contrast function g  
    >>>                 wg = np.tanh(ws * alpha).T


    >>>                 # Pass w*s into g'  
    >>>                 wg_ = (1 - np.square(np.tanh(ws))) * alpha


    >>>                 # Update weights  
                        wNew = (signals * wg.T).mean(axis=1) -   
    >>>                         wg_.mean() * w.squeeze()


    >>>                 # Decorrelate weights                
    >>>                 wNew = wNew -   
                               np.dot(np.dot(wNew, W[:c].T), W[:c])  
    >>>                 wNew = wNew / np.sqrt((wNew ** 2).sum())


    >>>                 # Calculate limit condition  
    >>>                 lim = np.abs(np.abs((wNew * w).sum()) - 1)  

    >>>                 # Update weights  
    >>>                 w = wNew  

    >>>                 # Update counter  
    >>>                 i += 1


    >>>             W[c, :] = w.T  
    >>>     return W

So now that we have all the code written up, lets run the whole thing!



    >>> # Center signals  
    >>> Xc, meanX = center(X)


    >>> # Whiten mixed signals  
    >>> Xw, whiteM = whiten(Xc)


    >>> # Run the ICA to estimate W  
    >>> W = fastIca(Xw,  alpha=1)


    >>> #Un-mix signals using W  
    >>> unMixed = Xw.T.dot(W.T)


    >>> # Subtract mean from the unmixed signals  
    >>> unMixed = (unMixed.T - meanX).T

The results of the ICA are shown in _Figure 4_ below where the upper panel
represents the original source signals and the lower panel the independent
components retrieved by our ICA implementation. And the result looks very
good. We got all three sources back!

![_config.yml]({{ site.baseurl }}/images/ICA/1d2sIJ40JDQ7Wa8-RIpM5dQ.png)
![_config.yml]({{ site.baseurl }}/images/ICA/1WvUqXPm9XrweP_KsQz8oVA.png)Figure
4: Results of the ICA analysis. Above true sources; below recovered signals.

So finally lets check one last thing: The kurtosis of the signals. As we can
see in _Figure 5_ all of our mixed signals have a kurtosis of ≤ 1 whereas all
recovered independent components have a kurtosis of 1.5 which means they are
less Gaussian than their sources. This has to be the case since the ICA tries
to maximize non-Gaussianity. Also it nicely illustrates the fact mentioned
above that the mixture of non-Gaussian signals will be more Gaussian than the
sources.

![_config.yml]({{ site.baseurl }}/images/ICA/1k1D5kIVqfJr8fdbsZJy0VQ.png)Figure
5: Kernel Density Estimates of the three mixed and source signals.

So to summarize: We saw how ICA works and how to implement it from scratch in
Python. Of course there are many Python implementations available that can be
directly used. However it is always advisable to understand the underlying
principle of the method to know when and how to use it. If you are interested
in diving deeper into ICA and learn about the details I recommend [this paper
by Aapo Hyvarinen and Erkki Oja,
2000](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf).

Otherwise you can check out the complete [code
here](https://github.com/akcarsten/Independent_Component_Analysis), follow me
on [Twitter](https://twitter.com/ak_carsten) or connect via
[LinkedIn](https://www.linkedin.com/in/carsten-klein/).

The code for this project can be found on
[Github](https://github.com/akcarsten/Independent_Component_Analysis).
