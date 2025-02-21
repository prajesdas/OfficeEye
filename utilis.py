import cv2
import numpy as np

# To make detections and get required outputs
def YOLO_Detection(model, frame, conf=0.10, mode = "track"):
    # Perform inference on an image
    if mode == "track":
        results = model.track(frame, conf=conf, iou = 0.1, classes = [0], persist=True)
        # Extract bounding boxes, classes, names, and confidences
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.cpu().tolist()
        ids = results[0].boxes.id.cpu().tolist()
        return boxes, classes, names, confidences, ids

    elif mode == "pred":
        results = model.predict(frame, conf=conf, classes = [0])
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.cpu().tolist()
        return boxes, classes, names, confidences


# Function to draw polygons and labels on a frame
def draw_working_areas(frame,area, index = int ,color = (113,179,60)):
    """
    Draws the defined working areas on the given frame.

    Parameters:
        frame (ndarray): The video frame on which to draw.
        areas (list): A list of working area coordinates.
    """

    # Convert area to a numpy array of shape (n_points, 1, 2)
    pts = np.array(area, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw the polygon
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    # Put the index number at the first vertex
    cv2.putText(frame, str(index + 1), (pts[0][0][0], pts[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


## Draw YOLOv8 detections function
def label_detection(frame, text, left, top, bottom, right, tbox_color=(152,251,152), fontFace=1, fontScale=0.7,
                    fontThickness=1):
    # Draw Bounding Box
    cv2.rectangle(frame, (int(left), int(top)), (int(bottom), int(right)), tbox_color, 1)
    # Draw and Label Text
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    y_adjust = 10
    cv2.rectangle(frame, (int(left), int(top) - text_h - y_adjust), (int(left) + text_w + y_adjust, int(top)),
                  tbox_color, -1)
    cv2.putText(frame, text, (int(left) + 5, int(top) - 5), fontFace, fontScale, (255, 255, 255), fontThickness,
                cv2.LINE_AA)

def drawPolygons(frame, points_list, detection_points=None, polygon_color_inside=(30, 205, 50),
                 polygon_color_outside=(30, 50, 180), alpha=0.5, occupied_polygons = int):

    for area in points_list:
        # Reshape the flat tuple to an array of shape (4, 1, 2)
        area_np = np.array(area, np.int32)
        if detection_points:
            is_inside = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points)
        else:
            is_inside = False
        color = polygon_color_inside if is_inside else polygon_color_outside
        if is_inside:
            occupied_polygons += 1

    return frame, occupied_polygons
