import cv2
import torch
import os

#this is my own thing. hopefully it doesnt mess anything up
if os.name == 'nt':
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
#end

# Load your trained YOLO model
model = torch.hub.load(r'C:\Users\user\Desktop\uni stuff\Swift\Swift-AI-Workspace\yolov5', 'custom', source='local', path=r"C:\Users\user\Desktop\uni stuff\Swift\Swift-AI-Workspace\useful-tools\1080p2-5zoom-test-3.pt")

# Open the video file
video_path = r"C:\Users\user\Desktop\uni stuff\Swift\Videos\swift-library-droppoints.mp4"
cap = cv2.VideoCapture(video_path)

# Define the codec and create a VideoWriter object to save the output video
#codec = cv2.VideoWriter_fourcc(*'XVID')
#output_video = cv2.VideoWriter('output_video.avi', codec, 30, (int(cap.get(3)), int(cap.get(4))))

frame_num = 0

while cap.isOpened():
    
    
    ret, frame = cap.read()
    if not ret:
        break
    #frame = cv2.resize(frame, (424, 240))
    # Perform object detection on the frame
    results = model(frame)
    print(results)
    print(frame.shape)
    resultsList = results.pandas().xyxy[0]#.to_numpy()
    print(len(resultsList))
    print(f'Frame #{frame_num}:', resultsList)

    if len(resultsList) >= 1:
        # Save the current frame under the name frame{frame_num}.png
        #cv2.imwrite(f'frame{frame_num}.png', frame)
        os.makedirs('images', exist_ok=True)
        cv2.imwrite(os.path.join('images', f'frame{frame_num}.png'), frame)
        # Convert the results to YOLO label format
        yoloFormattedLabel = ""
        for _, row in resultsList.iterrows():
            x_center = (row['xmin'] + row['xmax']) / (2 * frame.shape[1])
            y_center = (row['ymin'] + row['ymax']) / (2 * frame.shape[0])
            width = (row['xmax'] - row['xmin']) / frame.shape[1]
            height = (row['ymax'] - row['ymin']) / frame.shape[0]
            yoloFormattedLabel += f"{int(row['class'])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        
        # Save a txt file of the same name as the saved frame with the contents of resultsList
        os.makedirs('labels', exist_ok=True)
        with open(os.path.join('labels', f'frame{frame_num}.txt'), 'w') as f:
            f.write(yoloFormattedLabel)





    # Display the frame 
    #cv2.imshow('Object Detection', frame)
    #output_video.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_num +=1

cap.release()
#output_video.release()
cv2.destroyAllWindows()


