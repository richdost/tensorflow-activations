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
    - Further work
      - tanh and leaky_relu issues
      - Equal initial weights
      - Learning rate graph

## To run 
```bash
> python3 mnist_cnn.py
```

## Output
The learning curve for each activation method can be viewed 
 - graphically in tensorboard - View [a sample here](./sample_tensorboard_learning_curve.txt)
 - textually in the terminal output - View [a sample here](./sample_terminal_output.txt)


To view tensorboard output first run the mnist_cnn.py, then tensorboard.

```bash
> tensorboard --logdir='summary'
```

