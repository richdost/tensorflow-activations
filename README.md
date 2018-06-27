# tensorflow-activator-comparator
Compares effectiveness of activation functions in tensorflow for learning using MNIST data.
Class project for Machine Learning in Tensorflow, UC Berkeley Extension.

## To run 
```bash
> python3 compare_activation_functions.py
```

## Output
The learning curve for each activation method can be viewed 
 - graphically in tensorboard - View [a sample here](./sample_tensorboard_learning_curve.png)
 - textually in the terminal output - View [a sample here](./sample_terminal_output.txt)
 - bar chart comparing efficacy of activation functions - View [a sample here](./comparison_bar_chart.png)


#### More Info In Tensorboard
To view tensorboard or other output first run, then do tensorboard.

```bash
> python3 compare_activation_functions.py
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
    - Why MNIST
    - Why compare activation functions
  - Algorithm
    - Written for visibility
    - Ramp and Swish
    - Tanh
    - Weights equal
    - Walk through
  - Output
    - Tensorboard
    - Learning speed comparison

