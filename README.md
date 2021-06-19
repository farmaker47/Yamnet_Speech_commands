# Yamnet Speech commands

This is an effort to classify speech commands with the Yamnet model.

This repository contains 4 different notebooks.

### First notebook (Plain_Convolution_Speech_commands.ipynb)

This effort is with the use of a simple convolutional neural network (CNN). Dataset audio files are converted to waveforms and then into spectograms. The spectograms are then fed into the Keras model. You can find this example also [here](https://www.tensorflow.org/tutorials/audio/simple_audio). 

### Second notebook (Yamnet_Speech_commands.ipynb)

In this notebook Yamnet model is used to classify Speech commands and there is an extensive use of TensorFlow Datasets library. Audio files are converted to waveforms and if someone is shorter that one second it is padded with zeros. Then waveforms are fed to the yamnet model which gives 3 outputs. We are specially interested in the embeddings output of the Yamnet model which is then fed into a simple Keras model with 2 Dense layers for final classification. The notebook provides code for testing audio files, convert and save the 2 models into one and finally generate the .tflite model and test it with the TensorFlow Lite Interpreter.

Note: In this notebook ReduceMean is applied to the output of the model after classification.

### Third Notebook (Yamnet_Plain_Speech_commands.ipynb)

Inside this notebook there is the same manipulation of audio files with the previous one but here we do not use Datasets library. Waveforms are created and fed into the Yamnet model. Here we get the mean of the embeddings output on 0 axis and this becomes the input of the final classification model. The notebook provides code for testing audio files, convert and save the 2 models into one and finally generate the .tflite model and test it with the TensorFlow Lite Interpreter.


