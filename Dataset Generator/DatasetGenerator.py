import argparse
import cv2
import numpy as np
import math
import random
import time
import os

shapes = {'circle': np.array([(round(50 + 50 * math.cos(t * math.pi/180)), round(50 - 50 * math.sin(t * math.pi/180))) for t in range(0, 360, 4)]),
          'semicircle': np.array([(round(50 + 50 * math.cos(t * math.pi/180)), round(75 - 50 * math.sin(t * math.pi/180))) for t in range(0, 181, 4)]),
          'quartercircle': np.array([(25, 75)] + [(round(25 + 75 * math.cos(t * math.pi/180)), round(75 - 75 * math.sin(t * math.pi/180))) for t in range(0, 91, 3)]),
          'triangle': np.array([(50, 0), (100, 75), (0, 75)]),
          'square': np.array([(13, 13), (87, 13), (87, 87), (13, 87)]),
          'rectangle': np.array([(25, 0), (75, 0), (75, 100), (25, 100)]),
          'trapezoid': np.array([(25, 25), (75, 25), (100, 75), (0, 75)]),
          'pentagon': np.array([(50, 0), (98, 35), (79, 90), (21, 90), (2, 35)]),
          'hexagon': np.array([(25, 7), (75, 7), (100, 50), (75, 93), (25, 93), (0, 50)]),
          'heptagon': np.array([(50, 0), (89, 19), (99, 61), (72, 95), (28, 95), (1, 61), (11, 19)]),
          'octagon': np.array([(31, 4), (69, 4), (96, 31), (96, 69), (69, 96), (31, 96), (4, 69), (4, 31)]),
          'star': np.array([(50, 0), (61, 35), (98, 35), (68, 56), (79, 90), (50, 69), (21, 90), (32, 56), (2, 35), (39, 35)]),
          'cross': np.array([(33, 0), (66, 0), (66, 33), (100, 33), (100, 66), (66, 66), (66, 100), (33, 100), (33, 66), (0, 66), (0, 33), (33, 33)])}

colors = {'white': (255, 255, 255, 255),
          'black': (0, 0, 0, 255),
          'gray': (128, 128, 128, 255),
          'red': (0, 0, 255, 255),
          'blue': (255, 0, 0, 255),
          'green': (0, 255, 0, 255),
          'yellow': (0, 255, 255, 255),
          'purple': (255, 0, 128, 255),
          'brown': (0, 100, 150, 255),
          'orange': (0, 128, 255, 255)}

alphanumerics = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

classes = []

# Generator for a blank RGBA image, used as the background our shapes will be drawn on
def blank(n):
    img = np.zeros((n, n, 4), np.uint8)
    return img


# Generate a target image with the specified charateristics in string form
def target(shape, shape_color, alphanum, alphanum_color):
    img = blank(100)
    cv2.fillPoly(img, [shapes[shape]], colors[shape_color])
    retval, _ = cv2.getTextSize(alphanum, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4)
    cv2.putText(img, alphanum, (int(50 - retval[0] / 2), int(50 + retval[1] / 2)), cv2.FONT_HERSHEY_DUPLEX, 1.5,
                colors[alphanum_color], 4)
    return img

orientations = {'N': 0, 'NE': -45, 'E': -90, 'SE': -135, 'S': 180, 'SW': 135, 'W': 90, 'NW': 45}

def writeClassesFile():
    f = open("./output/labels.txt", "w")

    out = ""
    for line in classes:
        out += f"{line}\n"

    f.write(out)
    print(f"txt File {args.num_images+1}/{args.num_images+1} saved as ./output/labels.txt")

def debug_show_bounding_boxes(field, target_positions, target_shapes, shape_colors, alphanums, alphanum_colors, image_shape):
    field_with_boxes = field.copy()

    for pos, shape, shape_color, alphanum, alphanum_color in zip(target_positions, target_shapes, shape_colors, alphanums, alphanum_colors):
        x, y = pos

        target_img = target(shape, shape_color, alphanum, alphanum_color)
        h, w, _ = target_img.shape
        half_h, half_w = int(h * args.scale), int(w * args.scale)

        # Apply the offset
        x_offset, y_offset = args.offset
        x += x_offset
        y += y_offset

        # Draw bounding box
        cv2.rectangle(field_with_boxes, (int(y - half_w), int(x - half_h)), (int(y + half_w), int(x + half_h)), (0, 255, 0), 2)

        # Define the shape's class name
        class_name = f"{shape}_{shape_color}_{alphanum}_{alphanum_color}"

        if(args.debugger):
            print("------------------------------------------ \n class center_x center_y width height")
            print(f"{class_name} {round(y/image_shape[1],6)} {round(x/image_shape[0],6)} {round(2*half_h/image_shape[1],6)} {round(2*half_w/image_shape[0],6)}")

        # Draw text box with the class name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
        cv2.putText(field_with_boxes, class_name, (int(y + half_w), int(x - half_h)), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    #cv2.imshow('Debug - Bounding Boxes', field_with_boxes)
    import matplotlib.pyplot as plt
    plt.imshow(field_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def queueNewClass(classname):
    if classes.__contains__(classname):
        return classes.index(classname)
    classes.insert(len(classes), classname)
    return classes.index(classname)

def writeAnnotationFile(field, target_positions, target_shapes, shape_colors, alphanums, alphanum_colors, image_shape, img_name, image_index):
    field_with_boxes = field.copy()
    file = ""

    for pos, shape, shape_color, alphanum, alphanum_color in zip(target_positions, target_shapes, shape_colors, alphanums, alphanum_colors):
        x, y = pos

        target_img = target(shape, shape_color, alphanum, alphanum_color)
        h, w, _ = target_img.shape
        half_h, half_w = int(h * args.scale), int(w * args.scale)

        # Apply the offset
        x_offset, y_offset = args.offset
        x += x_offset
        y += y_offset

        #Mub: this is the place where the class name is being declared.
        #class_name = f"{shape} {shape_color} {alphanum} {alphanum_color}"
        class_name = f"object"  #this edited line makes it so there is only 1 class called object.
        #class_name = f"{shape}"
        classNo = queueNewClass(class_name)

        # DO NOT TOUCH THIS LINE OF CODE 
        # i will find you.
        # Mub: this is what generates the Yolov5 formatted labels/annotations for the images.
        file += f"{classNo} {round(y/image_shape[1],6):.6f} {round(x/image_shape[0],6):.6f} {round(2*half_h/image_shape[1],6):.6f} {round(2*half_w/image_shape[0],6):.6f}\n"

    f = open(f"{img_name[:-4]}.txt", "w")
    f.write(file)
    print(f"txt File {image_index + 1}/{args.num_images+1} saved as {img_name[:-4]}.txt")
        



def place_target(target_img, orientation, position, scale, existing_positions):
    alpha_channel = target_img[:, :, 3]
    target_img = add_noise(target_img)
    transformed_img = affine_transform(target_img, orientations[orientation], scale)
    transformed_alpha = affine_transform(alpha_channel, orientations[orientation], scale)

    field = alpha_blend(transformed_img, transformed_alpha, position)
    field = blur_target_edges(field, position, target_img, existing_positions)

    return field


def affine_transform(img, rotation, scale):
    rotation_matrix = cv2.getRotationMatrix2D((50, 50), rotation, 1)
    scale_matrix = cv2.getRotationMatrix2D((0, 0), 0, scale)

    new_dsize = (round(img.shape[0] * scale), round(img.shape[1] * scale))
    transformed_img = cv2.warpAffine(img, rotation_matrix, img.shape[:2])
    transformed_img = cv2.warpAffine(transformed_img, scale_matrix, new_dsize)
    return transformed_img

def add_noise(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img[row, col, 1] *= args.lighting_constant
            for i in [1, 2]:
                value = img[row, col, i]
                img[row, col, i] += min(255 - value, max(-int(value), random.randint(-args.noise_intensity, args.noise_intensity)))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def alpha_blend(img, alpha_channel, offset):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if alpha_channel[row, col] > 0:
                field[row + offset[0], col + offset[1], :3] = img[row, col]
    return field

def blur_target_edges(field, position, target_img, existing_positions):
    target_region = field[
        position[0]: position[0] + target_img.shape[0],
        position[1]: position[1] + target_img.shape[1]
    ]

    for existing_pos in existing_positions:
        distance = math.sqrt((position[0] - existing_pos[0])**2 + (position[1] - existing_pos[1])**2)
        blur_threshold = 50

        if distance < blur_threshold:
            return field

    blurred_target_region = cv2.GaussianBlur(target_region, (3, 3), 1)
    #blurred_target_region = cv2.GaussianBlur(target_region, (1, 1), 0.1)
    field[
        position[0]: position[0] + target_img.shape[0],
        position[1]: position[1] + target_img.shape[1]
    ] = blurred_target_region

    return field

folderMode = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num_images', type=int, default=100, help='The number of images to generate.')
    parser.add_argument('-t', '--num_targets', type=int, default=5, help='The number of targets to place in each mock field.')
    parser.add_argument('-s', '--scale_target', type=float, default=0.3, help='The average scale factor for each target.')
    parser.add_argument('-sv', '--scale_variance', type=float, default=0.2, help='The multiplication factor by which the scale of a single target can vary. Set to 0 for a constant scale.')
    parser.add_argument('-l', '--lighting_constant', type=float, default=0.5, help='The amount to scale each pixel saturation by, simulating natural lighting.')
    parser.add_argument('-n', '--noise_intensity', type=int, default=10, help='The maximum increase or decrease applied to HSV values in random noise generation.')
    parser.add_argument('-c', '--clip_maximum', type=float, default=0, help='The greatest proportion of a target\'s width/height that may be out of bounds. Zero by default, but set higher to allow clipping.')
    parser.add_argument('-d', '--debugger', type=bool, default=False, help='show bounding boxes (YOLO)')
    parser.add_argument('-o', '--offset', nargs=2, type=int, default=[15, 15], help='(CONTACT HAMZA) how much to the [right, down] should we move the bounding boxes')
    parser.add_argument('-sc', '--scale', type=float, default=0.2, help='(CONTACT HAMZA) how much should we scale the bounding boxes')
    parser.add_argument('-in', '--image_name', type=str, default="maryland_test.png")
    parser.add_argument('-f', '--image_folder', type=str, default="", help='the folder where the input images are stored')
    
    args = parser.parse_args()

    if args.image_folder != "":
        print(f"FOLDER MODE: WILL USE ALL BACKGROUND IMAGES FROM {args.image_folder}")
        fields = []
        for path in os.listdir(args.image_folder):
            if os.path.isfile(os.path.join(args.image_folder, path)):
                fields.append(path)
                folderMode = True

    else:
        fields = [args.image_name]

    output_directory = './output/'  

    os.makedirs(output_directory, exist_ok=True)

    for field_name in fields:
        for image_index in range(args.num_images):
            args = parser.parse_args()
            current_time_ms = int(time.time() * 1000)
            seed = random.randint(1, 1000000)*current_time_ms
            random.seed(seed)

            if folderMode:
                field = cv2.imread(os.path.join(args.image_folder, field_name))
                print(f"------------------------{field_name}------------------------")
            else:
                field = cv2.imread(field_name)

            image_shape = field.shape

            all_target_positions = []
            all_target_shapes = []
            all_target_shape_colors = []
            all_target_alphanums = []
            all_target_alphanum_colors = []

            existing_positions = []

            for i in range(args.num_targets):
                shape = random.choice(list(shapes.keys()))
                alphanum = random.choice(alphanumerics)
                shape_color, alphanum_color = random.sample(list(colors.keys()), 2)

                orientation = random.choice(list(orientations.keys()))
                scale = random.uniform((1-args.scale_variance)*args.scale_target, (1+args.scale_variance)*args.scale_target)

                # Check for overlapping and generate a new position if necessary
                while True:
                    pos = (
                        round(random.uniform(-args.clip_maximum*100*scale, field.shape[0]-(1-args.clip_maximum)*100*scale)),
                        round(random.uniform(-args.clip_maximum*100*scale, field.shape[1]-(1-args.clip_maximum)*100*scale))
                    )
                    bounding_box = (
                        pos[0] - int(50 * scale),
                        pos[1] - int(50 * scale),
                        pos[0] + int(50 * scale),
                        pos[1] + int(50 * scale)
                    )

                    overlap = False
                    for existing_pos in existing_positions:
                        existing_box = (
                            existing_pos[0] - int(50 * scale),
                            existing_pos[1] - int(50 * scale),
                            existing_pos[0] + int(50 * scale),
                            existing_pos[1] + int(50 * scale)
                        )
                        if (
                            bounding_box[2] > existing_box[0] and
                            bounding_box[0] < existing_box[2] and
                            bounding_box[3] > existing_box[1] and
                            bounding_box[1] < existing_box[3]
                        ):
                            overlap = True
                            break

                    if not overlap:
                        break

                existing_positions.append(pos)

                transformed_target = place_target(target(shape, shape_color, alphanum, alphanum_color), orientation, pos, scale, existing_positions)

                all_target_positions.append(pos)
                all_target_shapes.append(shape)
                all_target_shape_colors.append(shape_color)
                all_target_alphanums.append(alphanum)
                all_target_alphanum_colors.append(alphanum_color)


            field = blur_target_edges(field, pos, transformed_target, all_target_positions)

            if(args.debugger):
                debug_show_bounding_boxes(field, all_target_positions, all_target_shapes, all_target_shape_colors, all_target_alphanums, all_target_alphanum_colors, image_shape)

            image_filename = os.path.join(output_directory, 'OUT_{0}{1}.png'.format(field_name[:-4], seed))
            cv2.imwrite(image_filename, field)
            print(f"Image {image_index + 1}/{args.num_images} saved as {image_filename}")
            writeAnnotationFile(field, all_target_positions, all_target_shapes, all_target_shape_colors, all_target_alphanums, all_target_alphanum_colors, image_shape, image_filename, image_index)
    writeClassesFile()