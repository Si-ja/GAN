from networks.Generator import Generator
from utils.device import activate_device
from utils.current_date import date_label

import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the framework of a model
    G = Generator()
    
    # Put the model on the device of user's choise
    device = activate_device()
    G = G.to(device)
    
    # Indicate the path where all the weights are saved
    G_path = r"models\G_conv.ckpt"
    
    # Load the weights into the model
    G.load_state_dict(torch.load(G_path))
    
    # Put the model into a state of generating information
    G.eval()
    
    # Generate random noise with 100 datapoints and create 1 image with it
    # The noise instance follows the ~formula: batch_size x laten_space x height x width
    # Think of it as a line made out of different colors painted downwards
    noise = torch.randn((1, 100)).view(-1, 100, 1, 1)
    
    # Load the data to the device as well
    noise = noise.to(device)
    
    # Generate an image
    gen_img = G(noise)
    
    # Show the image by also transforming it to the shape the computer can work with
    # Turn off the default matplotlib axis values
    with torch.no_grad():
        img = gen_img.detach().clone().cpu().reshape(28, 28)
   
    # Visualize the image
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis("off")
    
    # Save the image by default to 'generated_examples' folder
    image_path = r"generated_examples\Generated Image at " + str(date_label()) + ".png"
    plt.savefig(fname=image_path)