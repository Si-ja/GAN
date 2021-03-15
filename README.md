# GAN
A simple GAN made with PyTorch. It is meant to work with 1 dimensional arrays
(i.e. black and white pictures, or more specifically grayscale).

As with regular GANs there are two components - the Generator and the Discriminator.
In the current instance they are both set up using basic concepts of how standard
Convolutional Neural Networks work.

Specifically with the repository, the code was developed to make a rough blueprint
of how GANs can be formed and later it is possible to expand on this.

The current code gives 2 possibilities to the user - train the networks from scratch
using their own criteria. And use the trained network to generate new images of
hand written digits.

# How to set up

I won't lie, the details are a bit funky, because I used the pre-existing environments
of mine. Therefore, the following requirements are an estimation on what you might
need, but should be enough to get you going

```bash
torch>=1.2.0
torchvision>=0.4.0
matplotlib>=3.1.1
```
Complications come from the fact that if you want to use an NVIDIA GPU you also
will have to perform installation of CUDA.

# How to train a new instance of networks

In a set up environment to operate with PyTorch and Matplotlib you can simply
execute the `train.py` script with needed parameters to train a newly created
instance of networks.

Run `python train.py -h` to see a list of all options you can pass to the
script to set up your own conditions for training of the networks. You should
be greated with the following information:

```bash
usage: train.py [-h] [-d DEVICE] [-lr LEARNING_RATE] [-e EPOCHS] [-b BATCH]
                [-s SAVE] [-sp SAVE_PATH] [-gn GENERATOR_NAME]
                [-dn DISCRIMINATOR_NAME]

-------------------------------------------
                DESCRIPTION:

A script that will train the neural networks
again, allowing to modify their behaviour.
-------------------------------------------
                REQUIREMENTS:

                >Python ~3.7
                >torch>=1.1.0
-------------------------------------------

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device to use for processing. cuda or cpu. Defaults to cpu
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate of the networks to operate. Default: 0.0002
  -e EPOCHS, --epochs EPOCHS
                        The amount of epoch networks learn through. Default: 5
  -b BATCH, --batch BATCH
                        The size of the data batches. Default: 32
  -s SAVE, --save SAVE  Save a new generated model after each epoch [y/n]. Default: y
  -sp SAVE_PATH, --save_path SAVE_PATH
                        Path where the models will be saved. Default: models\
  -gn GENERATOR_NAME, --generator_name GENERATOR_NAME
                        Name of the generator model. Default: G_net
  -dn DISCRIMINATOR_NAME, --discriminator_name DISCRIMINATOR_NAME
                        Name of the discriminator model. Default: D_net
```

Example of a command with which you can train the networks: `python train.py -d cuda -e 10 - b 16
-s y -lr 0.0002`. This would indicate that you want to use your GPU to train the
networks for 10 epoch, with a batch size of 16 for data, learning rate of 0.0002 and
save the models' checkpoints after each epoch ends.

Development
It is possible to generate images, as ones present in the generated examples.
For that you need to run the `generate.py` in the main folder. Without the 
describtion what tools are being used for this (i.e. modules) it might be more
difficult, however, if you have PyTorch installed and matplolib, you should be
able to work with the given approach. A more detailed set of instructions is to
come.

# How to make generated images with the network

New images can be generated using the `generate.py` file. After you initiate
the working environment with all of the needed packages, you can run a script
`python generate.py -h` to see in what way you can create new images. You
should be greated with the following screen:

```bash
usage: generate.py [-h] [-d DEVICE] [-m MODEL] [-sp SAVE_PATH] [-sn SAVE_NAME]

-------------------------------------------
                DESCRIPTION:

This file allows to generate hand-written
like grayscale images, similar to ones that
can be observed with the MNIST database.

Several parameters can be passed to the
script, in order to tailor the generation
of images to your liking.

The arguments developed do not heaviliy
verify if what you pass into them is viable.
Meaning performance of some commands might be
unstable and it is recommended for the users
to stick to default arguments.
-------------------------------------------
                REQUIREMENTS:

                >Python ~3.7
                >torch>=1.1.0
                >Matplotlib>=3.1.0
-------------------------------------------

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device to use for processing. cuda or cpu. Defaults to cpu
  -m MODEL, --model MODEL
                        path to the model to use for the network. Defaults to: models\G_conv.ckpt
  -sp SAVE_PATH, --save_path SAVE_PATH
                        path to save the image. Defaults to: generated_examples\
  -sn SAVE_NAME, --save_name SAVE_NAME
                        Name under which the generated image will be saved. Defaults to TIME and will save with the date and time of the image generation
```

Therefore, you can pass a command such as `python generate.py -d cpu -m models\G_net.ckpt
-sp "" -sn test` and  for this case in the base directory you will get an image
called _Generated Image test.png_ processed on a cpu with the network G_net saved
in the models folder. In my case I the image to the __generated_examples__ folder
to retain it for demonstration purposes.

# Examples of Good and Not so Good generations

![](https://github.com/Si-ja/GAN/blob/f4466dbe8551b199fc60c3bf1d98b0fa74824109/generated_examples/Generated%20Image%20at%2014_03_2021%2017_28_24.png "Clear 5")
![](https://github.com/Si-ja/GAN/blob/9be228c5bb1e55c146e83cf6c38d0f1072eeb793/generated_examples/Generated%20Image%20at%2014_03_2021%2017_28_28.png "Appears to be 8")
![](https://github.com/Si-ja/GAN/blob/9be228c5bb1e55c146e83cf6c38d0f1072eeb793/generated_examples/Generated%20Image%20at%2014_03_2021%2017_28_23.png "Fancy 2 or messed up 8")
