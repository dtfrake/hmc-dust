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

7/6

We can think of the momentum variable in HMC as "noise" in the overall potential energy V(q) instead of noise in just the position q. Adding noise in V(q) allows us to efficently explore ALL of the typical set (we are exploring a hypershell-shaped area centered at the mode instead of a small spherical area centered at q that misses most of the typical set)

-Update the HMC class to return ALL the intermediate position steps

-In HMC.run, you can add a keyword return_samples = True

-Check to see if KL divergence increases linearly with number of dimensions

-In lots of field inference problems, the chi_2 values using the posterior and the prior are both 1. Why? First of all, the chi_2 value using the posterior should (almost) always be 1 because, given that the prior is not unreasonably strong, the posterior should fit the data. If we let our sample space have way more dimensions (like 100000) than the number of data points we have, then the data
mathematically cannot constrain our prior distribution that much because it only restricts a small number of directions

-Better checks than just the KL divergence for a multivariate Gaussian: plot the diagonal of the unnormalized covariance matrix and compare with analytical posterior mean 
-plot the mean of the samples and compare with the analytical posterior mean
plot the KL divergence as a function of step size

-Task 7/8: Integrate P(d|s, f)P(f) to see if I get the same thing as P(d|s) with brute force (instead of the slick method showing it's a Gaussian with mean 0 and covariance RR^T + N)
-Task 7/8: Try to write out what formulas HMC will allow us to calculate 

P(xi | d) = P(d | xi)P(xi)/P(d)

With HMC, we are able to take samples xi from the posterior. So, the integral of f(xi)P(xi | d)dxi is roughly sum of f(xi_sample)

Plot the marginal scale distrubtion from the HMC scale inference samples and then bin it to directly compute KL divergence
Plot KL divergence vs step size again, but in every HMC sample vary the number of integration steps

7/8: Next Steps: Use the RBF kernel to do inference on a 1D Gaussian process by Fourier transforming it. 
My prior will be a unit Gaussian; I will have parameters in white noise. My reponse will include the Fourier Transform. I can take the Jacobian of it to get R

Once I do this, it will be easy to transform to Matern. 
Other tasks:

-Perfect fitting scale
-Play around with JVP and VJP instead of computing R direclty
-Try to do inference on integrated/exponentiated dust 

-Once I can do inference using a Matern kernel and fft, infter all three of the scale parameters (p, v, sigma) and then make marginal plots of them

7/13

-Plot real/imiginary parts of hartley transformation in 1d and confirm that you can understand the symmetry
-NIFTY when it treats the variance as fixed can generate a random unnormalzied GP, divide by the integral of the power spectrum, and then mulitply by what the variance
should be. This is useful when NIFTY is doing inference on flexible power spectrum. 

-Plot the marginal scale distribution of my RBF scale inference and test KL divergence (done! it also works a lot better now)
-Do the Matern inference, draw fake data from some actual matern, see if the heatmaps go around the truth
    -If I shrink the error bars on the data and add more data points, the posterior should be able to perfectly recover what the truth actually is
-Analytically grid the posterior


7/14

-Plot: Where is the Hessian positive semidefinite?
-Plot: The eigenvectors of the Hessian (at some arbitrary sampled points) done!
-Test RMHMC: fix the Hessian 
-Fisher metric: the "expected curvature" of the likelihood