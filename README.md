[The computer vision task](https://gist.github.com/csaftoiu/9fccaf47fd8f96cd378afd8fdd0d63c1) is to create a neural network which takes an image of a face as input and returns its orientation - upright, rotated_left, rotated_right, or upside_down - and use this neural network to correct the images in the test set.

## Dependencies
#### OpenCV
#### Pandas
#### Numpy
#### Scit-learn
#### Keras

## Neural Network utilized:

### [Pure Convolutional Neural Network](https://arxiv.org/pdf/1412.6806.pdf)



## Usage

Unzip all images in the directory

## Recreate model
```
$ python pure_convolutional.py
```

## Rotate Downloaded files
```
$ python change_images.py
```

# Results
in 20 epochs the model reached 96% accuracy , without overfitting :)

![Alt text](https://i.imgur.com/DHoTzgO.png)


