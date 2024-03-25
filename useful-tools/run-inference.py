import yolov5
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# load pretrained model
model = yolov5.load(pathlib.Path('useful-tools/best.pt'))

# or load custom model
#model = yolov5.load('train/best.pt')
  
# set model parameters
model.conf = 0.1  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
#img = r'C:\Users\Administrator.MININT-H2P38Q5\Desktop\uni stuff\Semester 3\code\python\testing\car.jpg'

#img = pathlib.Path('useful-tools/test-images/test3.jpg')
img = pathlib.Path('useful-tools/test-images/test3.jpg')

# perform inference
results = model(img)

# inference with larger input size
#results = model(img, size=1280)

# inference with test time augmentation
#results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# show detection bounding boxes on image
results.show()

# save results into "results/" folder
#results.save(save_dir='results/')


#r'C:\Users\Administrator.MININT-H2P38Q5\Desktop\uni stuff\Swift\Dataset Generator\output\OUT_road123101446688814685.png
#