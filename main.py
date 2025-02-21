import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection, draw_working_areas


def setup_device():
    """Check if CUDA is available and set the device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def load_yolo_model(device):
    """Load the YOLO model and configure it."""
    model = YOLO("yolov8n.pt")
    model.to(device)
    model.nms = 0.5
    print(f"Model classes: {model.names}")
    return model


def initialize_variables(num_areas):
    """Initialize time tracking variables."""
    time_in_area = {index: 0 for index in range(num_areas)}
    entry_time = {}
    return time_in_area, entry_time


def process_frame(model, frame, working_area, time_in_area, entry_time, frame_cnt, frame_duration):
    """Process a single frame to detect objects and track time spent in each area."""
    boxes, classes, names, confidences, ids = YOLO_Detection(model, frame, conf=0.05, mode="track")
    polygon_detections = [False] * len(working_area)

    for box, cls, id in zip(boxes, classes, ids):
        center_point = calculate_center(box)
        label_detection(frame=frame, text=f"{names[int(cls)]}, {int(id)}", tbox_color=(255, 144, 30), left=box[0],
                        top=box[1], bottom=box[2], right=box[3])

        for index, pos in enumerate(working_area):
            if cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False) >= 0:
                polygon_detections[index] = True
                track_time(id, index, frame_cnt, entry_time, time_in_area, frame_duration)

    draw_polygons(frame, working_area, polygon_detections)
    display_time_overlay(frame, time_in_area)


def calculate_center(box):
    """Calculate the center of a bounding box."""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)


def track_time(id, index, frame_cnt, entry_time, time_in_area, frame_duration):
    """Track time spent by each object in each working area."""
    if id not in entry_time:
        entry_time[id] = (frame_cnt, index)
    else:
        start_frame, area_index = entry_time[id]
        if area_index != index:
            time_in_area[area_index] += frame_duration
            print(f"Object ID {id} left Area {area_index + 1}. Time counted: {time_in_area[area_index]:.2f}s")
            entry_time[id] = (frame_cnt, index)
        else:
            time_in_area[area_index] += frame_duration
            if area_index == 5:
                print(f"Object ID {id} is in Area 6. Time counted: {time_in_area[area_index]:.2f}s")


def draw_polygons(frame, working_area, polygon_detections):
    """Draw working areas with specific color coding based on detections."""
    for index, pos in enumerate(working_area):
        color = (0, 255, 0) if polygon_detections[index] else (0, 0, 255)
        draw_working_areas(frame=frame, area=pos, index=index, color=color)


def display_time_overlay(frame, time_in_area):
    """Overlay time spent in each area on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 250), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    for index, time_spent in time_in_area.items():
        cv2.putText(frame, f"Cabin {index + 1}: {round(time_spent)}s", (15, 30 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def main(source_video):
    device = setup_device()
    model = load_yolo_model(device)
    working_area = [
        [(499, 41), (384, 74), (377, 136), (414, 193), (417, 112), (548, 91)],
        [(547, 91), (419, 113), (414, 189), (452, 289), (453, 223), (615, 164)],
        [(158, 84), (294, 85), (299, 157), (151, 137)],
        [(151, 139), (300, 155), (321, 251), (143, 225)],
        [(143, 225), (327, 248), (351, 398), (142, 363)],
        [(618, 166), (457, 225), (454, 289), (522, 396), (557, 331), (698, 262)]
    ]
    time_in_area, entry_time = initialize_variables(len(working_area))
    frame_duration = 0.1

    cap = cv2.VideoCapture(source_video)
    frame_cnt = 0

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter("E:\Office-Monitoring-with-ai-yolov8-master\output_video\work.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))  # Adjust the frame rate and size

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1

        process_frame(model, frame, working_area, time_in_area, entry_time, frame_cnt, frame_duration)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()  # Release the VideoWriter
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(source_video="E:\Office-Monitoring-with-ai-yolov8-master\input_video\work-desk.mp4")
