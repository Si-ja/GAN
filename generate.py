import sys
import argparse
from argparse import RawTextHelpFormatter

from networks.Generator import Generator
from utils.device import activate_device
from utils.current_date import date_label

import torch
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="-------------------------------------------\n"
                                                 "\t\tDESCRIPTION:\n\n"
                                                 "This file allows to generate hand-written\n"
                                                 "like grayscale images, similar to ones that\n"
                                                 "can be observed with the MNIST database.\n"
                                                 "\n"
                                                 "Several parameters can be passed to the\n"
                                                 "script, in order to tailor the generation\n"
                                                 "of images to your liking.\n"
                                                 "\n"
                                                 "The arguments developed do not heaviliy\n"
                                                 "verify if what you pass into them is viable.\n"
                                                 "Meaning performance of some commands might be\n"
                                                 "unstable and it is recommended for the users\n"
                                                 "to stick to default arguments.\n"
                                                 "-------------------------------------------\n"
                                                 "\t\tREQUIREMENTS:\n\n"
                                                 "\t\t>Python ~3.7\n"
                                                 "\t\t>torch>=1.1.0\n"
                                                 "\t\t>Matplotlib>=3.1.0\n"
                                                 "-------------------------------------------",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="Device to use for processing. cuda or cpu. Defaults to cpu")
    parser.add_argument("-m", "--model", type=str, default="models\\G_conv.ckpt",
                        help="path to the model to use for the network. Defaults to: models\G_conv.ckpt")
    parser.add_argument("-sp", "--save_path", type=str, default=r"generated_examples\\",
                        help="path to save the image. Defaults to: generated_examples\\")
    parser.add_argument("-sn", "--save_name", type=str, default=r"TIME",
                        help="Name under which the generated image will be saved. Defaults to TIME and will save with the date and time of the image generation")
    args = parser.parse_args()
    sys.stdout.write(str(generate(args)))
    return
    
def generate(args):
    print("[X] The process has started")
    # Load the framework of a model
    G = Generator()
    print("[X] The network loaded")
    
    # Put the model on the device of user's choise
    device = activate_device(device=args.device)
    G = G.to(device)
    print(f"[X] Activated {device} as a device to process generations")
    
    # Indicate the path where all the weights are saved
    G_path = args.model
    
    # Load the weights into the model
    G.load_state_dict(torch.load(G_path))
    print("[X] The networks' weights loaded")
    
    # Put the model into a state of generating information
    G.eval()
    
    # Generate random noise with 100 datapoints and create 1 image with it
    # The noise instance follows the ~formula: batch_size x laten_space x height x width
    # Think of it as a line made out of different colors painted downwards
    noise = torch.randn((1, 100)).view(-1, 100, 1, 1)
    
    # Load the data to the device as well
    noise = noise.to(device)
    
    # Generate an image
    with torch.no_grad():
        gen_img = G(noise)
    print("[X] The image generated")
    
    # Show the image by also transforming it to the shape the computer can work with
    # Turn off the default matplotlib axis values
    img = gen_img.detach().clone().cpu().reshape(28, 28)
        
    # Visualize the image
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis("off")
    
    # Save the image by default to 'generated_examples' folder
    image_path = args.save_path
    image_name = args.save_name
    if image_name == "TIME":
        image_name = str(date_label()) + ".png"
    save_path = image_path + "Generated Image " + image_name
    plt.savefig(fname=save_path)
    print("[X] The image has been saved.")
    
if __name__ == "__main__":
    _ = main()