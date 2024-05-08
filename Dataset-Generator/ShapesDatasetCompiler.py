import argparse
import time
import os
import subprocess



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num_images', type=int, default=1, help='The number of images to generate.')
    #parser.add_argument('-t', '--num_targets', type=int, default=1, help='The number of targets to place in each mock field.')
    #parser.add_argument('-s', '--scale_target', type=float, default=0.3, help='The average scale factor for each target.')  #this values is ignored, I overwrite it later on using this function: calculate_s_sc()
    #parser.add_argument('-sv', '--scale_variance', type=float, default=0.5, help='The multiplication factor by which the scale of a single target can vary. Set to 0 for a constant scale.')
    #parser.add_argument('-l', '--lighting_constant', type=float, default=0.5, help='The amount to scale each pixel saturation by, simulating natural lighting.')
    #parser.add_argument('-n', '--noise_intensity', type=int, default=10, help='The maximum increase or decrease applied to HSV values in random noise generation.')
    #parser.add_argument('-c', '--clip_maximum', type=float, default=0, help='The greatest proportion of a target\'s width/height that may be out of bounds. Zero by default, but set higher to allow clipping.')
    #parser.add_argument('-d', '--debugger', type=bool, default=False, help='show bounding boxes (YOLO)') #bruh. If you include this flag in your argument, then the value of it will always be true. to keep it false, make sure you dont add it in the command line argument. I think there is a way to make it so that if I pass the value of False, it gets set to False; but that is left to do for some other day.
    #arser.add_argument('-o', '--offset', nargs=2, type=int, default=[-1, -1], help='(CONTACT HAMZA) how much to the [down, right] should we move the bounding boxes')
    #parser.add_argument('-sc', '--scale', type=float, default=0.2, help='(CONTACT HAMZA) how much should we scale the bounding boxes') #this values is ignored, I overwrite it later on using this function: calculate_s_sc()
    #parser.add_argument('-in', '--image_name', type=str, default="maryland_test.png")
    parser.add_argument('-f', '--image_folder', type=str, default="./input-images/input-images-1", help='the folder where the input images are stored')
    parser.add_argument('-dim','--image_dimension', type=int, default=240, help='The (height) dimension of the output images that you want to generate. The width will be auto generated using 16:9 ratio. The input images will be resized to achieve this.' )
    parser.add_argument('-z', '--zoom_level', type=float, default=50, help='The zoom level that the camera will be operating at, when looking at the shapes from above.')
    #parser.add_argument('-od', '--output_dir', type=str, default='./output/', help='The directory where the output images, their labels, and the class list will be stored.')
    #parser.add_argument('-sh', '--shape', type=str, default='circle', help='The shape that you want to be on the output images')


    args = parser.parse_args()
    script_path = './/ShapesDatasetGenerator.py'

    shapes = ['circle', 'semicircle', 'quartercircle', 'triangle', 'rectangle', 'pentagon', 'cross', 'star']


    for shape in shapes:
        myod = ".//outputShapes//" + shape
        #print(myod)

        arguments = ['-od', myod, '-i', args.num_images, '-sh', shape, '-dim', args.image_dimension, '-f', args.image_folder, '-z', args.zoom_level]  # Example arguments to pass

        if os.name == 'nt':
            subprocess.run(['python', script_path] + [str(arg) for arg in arguments])
        else:
            subprocess.run(['python3', script_path] + [str(arg) for arg in arguments])

