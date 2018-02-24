## Introduction
This repo keeps the projects which were taught in the book, [*deep learning
with Python*]().

## Concepts
In this section, we summarize the concepts which I found very insightful in the
book.

*machine learning*: "searching for a useful representations of some input data,
within a predefined space of possibilities, using guidance from a feedback
signal."

*loss function (objective function)*: the quantity that will be minimized during
training. It represents a measure of success for the task at hand.

*optimizer*: determine how the network (weights) will be updated based on the
loss function. It implements a specific variants of stochastic gradient descent
(SGD).


### Variant of Neural Network architect

**Sequential**: a linear stack of layers

**Multi-input models**:

**Multi-output models**:

**DAG**: Directed Acyclic Graph of Layers (e.g. Inception)

**Challenges**:
    * gradient vanishing

### Advanced Keras/Tensorflow Techniques
**callback** object which could turn the airplane of training model, sending it
off and lose the control over its trajectory, into autonomous plane where it
sense the environment and react to the changes.

Examples to use callbacks:
    * Model checkpoint -- saving the current weights of the model at different
    points during training
    * Early stopping -- Interrupting training whne the validation loss is no longer
    improving.
    * Dynamically adjusting the value of certain parameters during training --
    Such as the learning rate of the optimizer.
    * Logging training and validation metrics during training, or visualizing
    representations learned by the model as they're updated -- The Keras progress
    bar that you are familiar with is a callback!

**API**
keras.callbacks.ModelCheckpoint
keras.callbacks.EarlyStopping
keras.callbacks.LearningRateScheduler
keras.callbacks.ReduceLROnPlateau
keras.callbacks.CSVLogger
keras.callbacks.TensorBoard
kears.callbacks.CallBack

### Hyperparameter Tuning
It is very challenging task to find the optimal hyper-parameters. The following
reasons:
* it is very expensive to collect feedback signal which requires training a model
from scratch.
* The set of hyper-parameters normally involves discrete values, which are not
differentiable and  could utilize effective gradient-based optimization algorithm
to tell which next parameters to try.
