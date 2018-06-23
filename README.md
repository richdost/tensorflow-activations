# tensorflow-activator-comparator
Compares effectiveness of activation functions in tensorflow for mnist cnn

## To run 
```bash
> python3 mnist_cnn.py
```

## Output

Note that tanh and sigmoid are not learning.
That may be because the clip is not correct for them.
Currently outputs something like the below... 

```bash

----- using relu -----
2018-06-23 00:50:13.386551: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Epoch  0  has distance 0.7328527103229007
Epoch  1  has distance 0.2805148254741321
Epoch  2  has distance 0.2121418838270687
Epoch  3  has distance 0.17210330872711804
Epoch  4  has distance 0.144241665960713
Trained to accuracy  0.966

----- using softmax -----
Epoch  0  has distance 2.7162891522320853
Epoch  1  has distance 2.3150981348211124
Epoch  2  has distance 2.0427601632204913
Epoch  3  has distance 1.5247709621082635
Epoch  4  has distance 1.3983862657980506
Trained to accuracy  0.6343

----- using tanh -----
Epoch  0  has distance 34.86574779770596
Epoch  1  has distance 35.11108828111129
Epoch  2  has distance 35.11108824643223
Epoch  3  has distance 35.11108819094576
Epoch  4  has distance 35.11108829845088
Trained to accuracy  0.098

----- using sigmoid -----
Epoch  0  has distance 1.4808614483746614
Epoch  1  has distance 0.4326761100779882
Epoch  2  has distance 0.3346412042596123
Epoch  3  has distance 0.2744778209856969
Epoch  4  has distance 0.23526862830600948
Trained to accuracy  0.959

----- using relu6 -----
Epoch  0  has distance 0.5465756025503977
Epoch  1  has distance 0.20535644328052347
Epoch  2  has distance 0.1497426909072833
Epoch  3  has distance 0.11501085244457829
Epoch  4  has distance 0.09178674485788421
Trained to accuracy  0.9755
Tiny:compare-activation-functions rd$ py mnist_cnn.py 
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz

----- using relu -----
2018-06-23 00:54:02.653046: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Epoch  0  has distance 0.7757766861536287
Epoch  1  has distance 0.2790066825530747
Epoch  2  has distance 0.21475251123309136
Epoch  3  has distance 0.18127289535308427
Epoch  4  has distance 0.14959372145885774
Epoch  5  has distance 0.1281348303705454
Epoch  6  has distance 0.10947015094570814
Epoch  7  has distance 0.10226850828663875
Epoch  8  has distance 0.09075027149339965
Trained to accuracy  0.9714

----- using softmax -----
Epoch  0  has distance 2.5766262292861932
Epoch  1  has distance 2.0173745493455373
Epoch  2  has distance 1.9068823014606113
Epoch  3  has distance 1.6740434609759929
Epoch  4  has distance 1.5335161768306385
Epoch  5  has distance 1.4959310052611603
Epoch  6  has distance 1.479751307747582
Epoch  7  has distance 1.2906898694688635
Epoch  8  has distance 0.9728587867996911
Trained to accuracy  0.7636

----- using tanh -----
Epoch  0  has distance 34.796989590471476
Epoch  1  has distance 34.919081011685485
Epoch  2  has distance 34.91908100128181
Epoch  3  has distance 34.919080952731065
Epoch  4  has distance 34.919081025557084
Epoch  5  has distance 34.91908098394224
Epoch  6  has distance 34.919081042896636
Epoch  7  has distance 34.919080952731065
Epoch  8  has distance 34.919080942327334
Trained to accuracy  0.1028

----- using sigmoid -----
Epoch  0  has distance 34.805275249047774
Epoch  1  has distance 34.97363654396754
Epoch  2  has distance 34.973636550903365
Epoch  3  has distance 34.97363658211451
Epoch  4  has distance 34.9736365439675
Epoch  5  has distance 34.973636537031766
Epoch  6  has distance 34.97363658558241
Epoch  7  has distance 34.9736365092885
Epoch  8  has distance 34.97363653703171
Trained to accuracy  0.101

----- using relu6 -----
Epoch  0  has distance 1.0451817233725043
Epoch  1  has distance 0.3613791104202922
Epoch  2  has distance 0.2866309686546975
Epoch  3  has distance 0.24008625377985576
Epoch  4  has distance 0.2030707496743312
Epoch  5  has distance 0.1825982018526307
Epoch  6  has distance 0.16130978942933386
Epoch  7  has distance 0.14649532072584726
Epoch  8  has distance 0.1346444921974431
Trained to accuracy  0.9677

```


