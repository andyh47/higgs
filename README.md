The Higgs Boson Challenge just wrapped up on Kaggle.com where I placed 98 out of 1785 teams competing to create a machine learning flow that correctly classifies Higgs Boson events.  Organized by CERN (the Large Hadron Collider people) and Kaggle, participants were given a training set of 250000 events and a test set of 550000 events. The data consisted of 30 variables describing the momentum and other characteristics of particles generated in high-energy proton-proton collisions.  Higgs Bosons are particularly hard to classify in this data set because the particles created by a Higgs decay are very similar to another interaction decay path.  There is a lot of noise and not much signal. This was a great Challenge, with a lot of competitors, a lot of discussion and a lot of lessons.  Here's the approach that I took.

**The Flow**

For the Challenge I used Python, iPython Notebook, Sci-kit Learn, MongoDB and 8-core and 32-core AWS EC2 Ubuntu servers (and my trusty 2008 MacBook Pro).   Like many others I used Darin Baumgartel's starter kit to get going.  Very nicely coded, I ended up retaining the weight normalization code and the submission csv output.  The preferred classification algorithm for many in the Challenge was xgboost, a-multi threaded gradient boosting algorithm that worked very well on this data set.  While I tried other algorithms such as Random Forest, xgboost worked best for me. The first step in the flow is to find the best parameters for xgboost using the Python's HyperOpt module.  Once a promising parameter set is found, that model is bagged 100 times to reduce model variance and the prediction results are stored in MongoDB documents.  The last step is to select several models from the database and combine then with a simple averaging stacker. Run times for a 5 model stack are about 4 hours on an 8-core machine.

The code for this flow is organized as follows (.py and .ipynb versions are available):
- data_prep-ln-2split:  split the training dataset into train and validation sets, split out weights etc.
- withHyperopt-xgboost:  Search for best xgboost parameters
- withBagging-xgboost:   Bag a good model 100 times and save results for stacking
- stackerAVG:   Simple averaging stacker
- hyper_lib:  functions from plotting hyper results
- higgs_lib:  functions for computing AMS, signal and noise true post ivies and negatives, etc.

**Feature Engineering**

I explored a variety of feature engineering options, mostly without success.  Combining or eliminating variables based on PCA or Random Forest Importance measures resulted in worse results both on my locally calculated metric and on the Public Leaderboard.  This data set was carefully constructed by physicists at CERN, which may explain this result.  If you're trying to advance the state of the art in high energy physics classification, which was the point of the Challenge, then seeding the data with useless parameters doesn't buy anything.  Since the signal and noise overlapped so heavily in some areas, I tried kernel PCA to transform the data set to a higher dimension hoping to separate signal from noise.  Unfortunately, this did not provide improvement but in the process, I found a recent paper (June 2014), describing an approximation to kernel PCA, called IdealPCA, that provides the results similar to kernel PCA but can run on large data sets using modest amounts of memory and time.  (See: Learning with Cross-Kernels and Ideal PCA, http://arxiv.org/abs/1406.2646).  The only available implementation at this time is Matlab/Octave which is what I used.  Should be a very useful algorithm.  In the end, there were only couple of feature changes that I applied: I log transformed a half dozen variables that had skewed distributions and added two dummy variables related to the presence or absence of so-called "jets" in the decay products.  

The Challenge got an eleventh-hour surprise when a team of physicists released a set of new features hours before the end of the challenge.  These were based on computations on the original data set guided by high-energy physics knowledge and ~5000 hours of computer time.  This lead to a lot of scrambling at the end and a lot of changes in the Leaderboard. I made a submission using this additional data but, although it ranked higher on the Public Leaderboard, in the end it was not as good as my best previous set.

**Lessons Learned**

The scoring metric for the Challenge is called Approximate Median Significance (AMS).  AMS was noted by many to be very unstable, depending highly on the exact set of data used to calculate it.  My approach to dealing with this was to bag heavily and stack to stabilize my results.  My predictions of AMS on the other hand used only a simple train and validation split.  This lead to local AMS scores which varied significantly from the Leaderboard scores.  This is a big disadvantage because with an unstable AMS you can't be sure that you're putting your best model forward.  My take-away is that it's worthwhile to spend as much time as necessary at the start in order to get a stable metric prediction.  For example, one Forum poster  used 4-fold cross validation, repeated 5 times to achieve stability and accuracy for AMS. Time-consuming but it improved accuracy and stability. 

Another improvement that I would make is to use a more sophisticated stacking methodology, e.g. ridge regression.  To avoid overfitting with this, I would combine cross validation with a hold-out validation set to be used only on the stacking algorithm.  Lastly, I would like to explore a more systematic way to do feature generation, for example using a neural network or deep learning to generate new features.  

Thanks to Kaggle and CERN for putting on the Challenge.



