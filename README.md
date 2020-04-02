# Neural Networks from scratch

This project is heavily inspired by the awesome e-book http://neuralnetworksanddeeplearning.com/.

Using just `numpy` we load the MNIST data directly in binary format from http://yann.lecun.com/exdb/mnist/, build the network, and train it using [momentum SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum). Optionally L1/L2 regularization can be performed during training, although I have obtained better results without it.

Aside from training the model and testing it, the code offers the following functionalities:

- loading images from bytes to `numpy` arrays.
- building arbitrary dense (deep) neural network models, which can be initialized in any way (uniformly in tests).
- saving and loading pre-existing model weights from `JSON` archives.
- plotting training data (accuracy, loss, weight changes L2 norms) with `matplotlib`.

Below I provide a summary of the main mathematical operations employed (mostly as notes for eventual future developments).

<object data="readme_assets/readme.pdf" type="application/pdf" width="1000px" height="700px">
    <embed src="readme_assets/readme.pdf">
        <p>This browser does not support PDFs. Please refer to <a href="readme_assets/readme.pdf">readme.pdf</a> directly.</p>
    </embed>
</object>