This script is used test whether human-recognizable images can be evolved using a Genetic Algorithm that leverages the softmax scores from a CNN as a fitness function.

#  Important Note

The goal was to determine if a Genetic Algorithm could be used in conjunction with a CNN to evolve human-recognizable images.  It turns out that this is NOT possible.  None the less, I chose to upload this project to GitHub, as I thought it was a novel, interesting idea that people might enjoy hearing about.

A video describing this project and the outcomes can be found [here](https://www.youtube.com/watch?v=3h3L5joNXyY).

## The Idea

I woke up in the middle of a plane ride with an idea:  If CNNs can be used to identify images, then using a CNN as the fitness function for a Genetic Algorithm should allow one to evolve new images.  For example, if a CNN classifies images of dogs, then a Genetic Algorithm can evolve an array of pixel values until it finds a combination that form an image that is classified convincingly as a dog.  Simple right?

Why does this matter?  This will allow us to generate images of things that don't actually exist.  Think about that for a moment.  Say you trained a CNN to recognize images of people's faces.  Now you use the Genetic Algorithm to evolve an image of a new face.  This face would not belong to any one person.  It would be a brand new representation of a human who never actually existed!  We would essentially be exploiting a CNNs ability to understand abstract concepts in image recognition to create new, never before seen things!  It's a bit like letting your CNN *imagine* something new!  How cool is that?!

##  Tech Details

The tech that I used in this project is as follows:

*  All CNNs were created by retraining TensorFlow's the Inception model.
*  The DEAP framework was used to create and run my Genetic Algorithm.
*  During this project, the CNNs were trained to try and classify images of 1's from the MNIST dataset.

Additional tech details:

* All images used in this experiment were converted from grayscale to simple black and white.
* The softmax values returned from the CNN were used as the fitness scores for the Genetic Algorithm.  Softmax values can be interpreted as probabilities.  Therefore, an image that is classified as a dog with a score of 0.99 has an extremely strong resemblance to a dog according to the CNN.
* Though I am trying to evolve images of 1's, classifiers must be trained to differentiate between multiple classes.  Therefore, what were the other classes on which my CNN was trained?  I tried a variety of CNNs that had all been trained under different circumstances.  Some of these attempts and the logic behind them are as follows:

  -  CNN trained on the MNIST dataset and could classify images as belonging in classes 0 through 9.
  -  CNN trained on images of 1's from MNIST and images of noise with 50% black and 50% white pixel composition.  The idea here was that we would want to evolve images to look less like random noise and more like an image of a 1.
  -  CNN trained on images of 1's from MNIST and images of noise with 95% black and 5% white pixel composition.  The idea here is that the ratio for pixel composition is closer to that in the images of the 1's.
  -  CNN trained on images of 1's from MNIST and images of pure black.  The idea here is that this is the simplest possible alternate class, and hopefully it will be easy to evolve away from a plane black background.
  -  CNN trained on images of 1's from MNIST and images of those same 1's with the colors inverted.  The idea here is that the two classes are literal opposites of one another.  In evolving an image of a 1, any pixels that are not explicitly correct will be actively be working towards the classification of the other class.  As such, a high fitness/softmax score in the configuration should be a strong indicator that a proper 1 has been drawn.
  -  Same idea as the previous bullet, but all the images of 1's (and its inversion) are of the same drawing of a 1.  The core logic here is the same as in the previous bullet, only now the dataset is even simpler.  I was hoping that this simplicity would increase the likelihood of success.  

## Results

There were two ways that this experiment could have gone:  Either the Genetic Algorithm would produce human-recognizable images with a high softmax score, or the Genetic Algorithm would produce nonsense-looking images with a high softmax score.  In either case, the images would look convincing to the CNN classifier, as that is what high softmax scores indicate.

Unfortunately, the images rendered were nonsense-looking.  In all CNN configurations, the same result was realized: None of the Genetic Algorithm trials yielded any images resembling an image of a 1.  In hindsight, the reason for this is obvious.  Classifiers work by learning distinguishing features between the classes.  Unfortunately, these features are not the same thing as the image of the thing itself.  For example, if your two classes were images of 0's and images of 1's, a classifier might learn that any curved line means 0, and then classify nonsense images with a curved line as a 0.  In our case, we essentially evolved images that contain these distinguishing features.  Because these features are abstracted, the results were unfortunately not human-recognizable.












