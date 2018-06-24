# tensorflow-activator-comparator
Compares effectiveness of activation functions in tensorflow for mnist cnn


## Presentation
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
The Ramp and Swish activation functions are not in tensorflow so were implemented.
They are very simple.
  - [Ramp](./ramp_activation_function.png)
  - [Swish](./swish_activation_function.png)

