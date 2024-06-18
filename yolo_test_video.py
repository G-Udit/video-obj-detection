import cv2
import torch
import json
import glob
import time


def run_inference(video_file_path,confidence_threshold,class_dict,output_path):
    start_time = time.time()
    # Define confidence threshold and class names (modify class_names for your dataset)
    conf_threshold = confidence_threshold
    with open('labels.txt','r') as f:
        class_names =  class_dict
    # Load YOLOv7 model (replace with your downloaded model path)
    model_path = "yolov7.pt"
    model = torch.hub.load('.', 'custom', 'yolov7.pt', source='local') 

    # Load video
    cap = cv2.VideoCapture(video_file_path)  # Replace with your video path

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video!")
        exit()

    # Define output video writer (modify codec and parameters if needed)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Initialize empty list for detections
    detections_data = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Break the loop if there are no frames captured
        if not ret:
            break

    # Convert frame to RGB (YOLOv7 expects RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference on the frame
        results = model(frame)
    #print(results)

    # Extract detected objects
        detections = results.pandas().xyxy[0]
        #print("?"*20,detections.head())

    # Loop through detections and draw bounding boxes (optional)
        for i in range(len(detections)):
            # Get confidence score and class label
            confidence = detections["confidence"][i]
            class_id = detections["class"][i]

            # Filter detections based on confidence threshold
            if confidence > conf_threshold:
                # Extract bounding box coordinates
                box = detections[["xmin", "ymin", "xmax", "ymax"]].iloc[i].tolist()

                # Convert coordinates to integers for OpenCV (optional)
                x_min, y_min, x_max, y_max = map(int, box)

                # Draw bounding box and class label on the frame (optional)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, class_names[class_id], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

        # Prepare detection data for JSON
        frame_data = []

        for i in range(len(detections)):
            if detections["confidence"][i] > conf_threshold:
                box = detections[["xmin", "ymin", "xmax", "ymax"]].iloc[i].tolist()
                frame_data.append({"class": class_names[class_id], "confidence": float(confidence), "bbox": box})
        detections_data.append(frame_data)

        # Display the resulting frame (optional)
        #cv2.imshow('Object Detection', frame)

        # Write frame to output video
        out.write(frame)

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture, destroy windows, and save detections to JSON
    cap.release()
    out.release()
    #cv2.destroyAllWindows()

    with open('detections.json', 'w') as f:
        json.dump(detections_data, f)

    end_time = time.time()
    print(f"Finished processing video. Detections saved to detections.json. The process took {end_time-start_time:.5f} seconds")
    return("Processing successful")

if __name__=="__main__":
    
    video_file_path = glob.glob("*mp4*")[0]
    confidence_threshold = 0.5
    with open('labels.txt','r') as f:
        class_dict =  dict(enumerate(f.read().split('\n')))
    
    output_path = "inference_"+video_file_path
    #print(output_path)
    run_inference(video_file_path,confidence_threshold,class_dict,output_path)
