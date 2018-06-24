# tensorflow-activator-comparator
Compares effectiveness of activation functions in tensorflow for mnist cnn.
Class project for Machine Learning in Tensorflow, UC Berkeley Extension.

## To run 
```bash
> python3 mnist_cnn.py
```

## Output
The learning curve for each activation method can be viewed 
 - graphically in tensorboard - View [a sample here](./sample_tensorboard_learning_curve.png)
 - textually in the terminal output - View [a sample here](./sample_terminal_output.txt)


To view tensorboard output first run the mnist_cnn.py, then tensorboard.

```bash
> tensorboard --logdir='summary'
```

## Ramp and Swish
These activation functions are not in tensorflow so were implemented in code.
They are very simple.
  - [Ramp](./ramp_activation_function.png) is implemented as
  ```python
  def ramp(x):
    return tf.clip_by_value(x, clip_value_min=0.001, clip_value_max=0.999)
  ```
  - [Swish](./swish_activation_function.png) is implemented as
  ```python
  def swish(x):
    return tf.nn.sigmoid(x) * x
  ```


## Presentation
Outline for class presentation
  - Introduction
    - Why NMIST
    - Why compare activation functions
  - Output
    - Tensorboard
    - Learning speed comparison
  - Algorithm
    - Low level so all visible
    - Walk through
      - Ramp and Swish
    - Further work
      - tanh etc issues (local minima?)
      - Equal initial weights
      - Learning rate graph

