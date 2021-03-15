import sys
import argparse
from argparse import RawTextHelpFormatter

from networks.Generator import Generator
from networks.Discriminator import Discriminator

from utils.device import activate_device
from utils.weights_init import weights_init
from utils.data_processor import generate_loader

import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="-------------------------------------------\n"
                                                 "\t\tDESCRIPTION:\n\n"
                                                 "A script that will train the neural networks\n"
                                                 "again, allowing to modify their behaviour.\n"
                                                 "-------------------------------------------\n"
                                                 "\t\tREQUIREMENTS:\n\n"
                                                 "\t\t>Python ~3.7\n"
                                                 "\t\t>torch>=1.1.0\n"
                                                 "-------------------------------------------",
                                     formatter_class=RawTextHelpFormatter)
    
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="Device to use for processing. cuda or cpu. Defaults to cpu")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0002,
                        help="The learning rate of the networks to operate. Default: 0.0002")
    parser.add_argument("-e", "--epochs", type=int, default=5,
                        help="The amount of epoch networks learn through. Default: 5")
    parser.add_argument("-b", "--batch", type=int, default=32,
                        help="The size of the data batches. Default: 32")
    parser.add_argument("-s", "--save", type=str, default="y",
                        help="Save a new generated model after each epoch [y/n]. Default: y")
    parser.add_argument("-sp", "--save_path", type=str, default="models\\",
                        help="Path where the models will be saved. Default: models\\")
    parser.add_argument("-gn", "--generator_name", type=str, default="G_net",
                        help="Name of the generator model. Default: G_net")
    parser.add_argument("-dn", "--discriminator_name", type=str, default="D_net",
                        help="Name of the discriminator model. Default: D_net")
    args = parser.parse_args()
    sys.stdout.write(str(train(args)))
    
def train(args):
    print("[X] Starting the process")
    # Create Networks
    G = Generator()
    D = Discriminator()
    print("[X] Networks Generated")
    
    # Set up a device to operate under
    device = activate_device(device=args.device)
    print(f"[X] Device {device} activated")
    
    # Move networks to the devices
    D = D.to(device)
    G = G.to(device)

    # Chose the criterion - which is Binary Classifier and create optimizers
    criterion = nn.BCELoss()
    lr = args.learning_rate
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

    # initialize generator and discriminator weights
    G.apply(weights_init)
    D.apply(weights_init)
    
    num_epochs = args.epochs
    batch_size = args.batch
    
    # Prepare the data to work with
    data_loader = generate_loader(data_path=r"utils\data", batch_size=batch_size, shuffle=True)
    print("[X] Data prepared")
    
    # Create a training structure
    total_steps = len(data_loader)

    # Put the networks into the training mode
    D.train()
    G.train()
    
    # Get last parameters before the trainings starts
    save_after_checkpoints = args.save
    save_path = args.save_path
    G_name = args.generator_name
    D_name = args.discriminator_name
    
    # Initiate the steps through epochs
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            mini_batch = images.size()[0]
            
            # Create labels with which we identify the data
            real_labels = torch.ones(mini_batch, 1).to(device)
            fake_labels = torch.zeros(mini_batch, 1).to(device)
            
            # !!!!!!!!!!!!!!!!!!!!!!! TRAIN THE DISCRIMINATOR !!!!!!!!!!!!!!!!!!!!!!!
            D.zero_grad()
            images = images.to(device)
            outputs = D(images.to(device))
            
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs
            
            # Computer the Binary Cross Entropy with fake images, that are generated by the Generator from random noise
            z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs
            
            # Calculate the value for backpropogation and optimize
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # !!!!!!!!!!!!!!!!!!!!!!! TRAIN THE GENERATOR !!!!!!!!!!!!!!!!!!!!!!!
            G.zero_grad()
            z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            
            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            g_loss = criterion(outputs, real_labels)
            
            # Backpropogation and optimization
            g_loss.backward()
            g_optimizer.step()
            
            if (i+1) % 200 == 0:
                log = (
                    f"Epoch: [{epoch+1}/{num_epochs}], "             \
                    f"Step: [{i+1}/{total_steps}], "                 \
                    f"D_loss: {round(d_loss.item(), 3)}, "           \
                    f"G_loss: {round(g_loss.item(), 3)}, "           \
                    f"D(x): {round(real_score.mean().item(), 3)}, "  \
                    f"D(G(z)): {round(fake_score.mean().item(), 3)}"
                )
                print(log)
        print("==================================================================================================")
        
        if save_after_checkpoints in ["y", "yes", "Y", "YES"]:
            # Save the model checkpoints 
            torch.save(G.state_dict(), save_path + G_name + ".ckpt")
            torch.save(D.state_dict(), save_path + D_name + ".ckpt")
            
    # Save models in any case if the full training finished
    torch.save(G.state_dict(), save_path + G_name + ".ckpt")
    torch.save(D.state_dict(), save_path + D_name + ".ckpt")
    print("[X] The network are fully prepared")

if __name__ == "__main__":
    main()