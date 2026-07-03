plot reduced chi_2
in HMC, you don't know what the distribution of observed counts is since we don't know the distrubtion of randomness in each bin
calculate KL divergence per degree of freedom
test by computing KL divergence of random samples drawn from the actual posterior dentisty with jax.random.choice
if you didn't have jax.random.choice, then flatten the array, compute the cdf, take a random real number between 0 and 1, and then take the cdf
Frequentist: fix a true data, get lots of fake data...



How to calculate chi_2 diagnostic given posterior samples (further reading: posterior predictive distribution)
First, draw a random posterior sample and pretend that it is the "real" data.
Then, calculate the chi^2 statistic of the measurements we took given what our "real" data is and the noise. For example, if our real
data is (0.5, 0.5) and our measurement is e^x + e^y = 1 +\- 0.1, the chi^2 would be ((e^0.5 + e^0.5 - 1)/0.1)^2. As another example,
if our "real" data is a Gaussian process described by a 50d vector d and we have 50 measurements with mean d' and noise 
covariance N, the chi^2 would be (d' - d)^T N^(-1) (d' - d)
#we can average over all the posterior samples. Note: if the chi^2 is really high, we could still be doing the inference correctly
but then the prior conflicts a lot with the data

We can also do the same against the prior 

On a .py file if you do # %% you can create a cell that VScode will run on its own

-Make my multidimensional HMC into its own module, test it on Gaussian process problem, perform tests 
    -KL divergence comparing a Gaussian fit of the HMC posterior to the actual analytically computed posterior (done!)
    -chi_2 test of "real" data against the measurement
    -chi_2 test of "real" data against the prior

Actually comptue the covariance matrix of my posterior samples for the bannana problem and see how it works in the mass matrix