import cv2
import numpy as np
from ultralytics import YOLO
from onvif import ONVIFCamera
import time
import threading
import onvif.exceptions
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load the YOLO model
model = YOLO("yolov8n-face.pt")

# RTSP stream configuration
rtsp_url = "rtsp://admin:admin123@192.168.1.7:554/stream2"
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    logging.error("Could not open video stream")
    exit()

# Video writer setup
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('multnew.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# ONVIF camera setup
camera = ONVIFCamera('192.168.1.7', 2020, 'admin', 'admin123')
media_service = camera.create_media_service()
ptz_service = camera.create_ptz_service()
media_profile = media_service.GetProfiles()[0]

# Camera dictionary
cameras = {
    'main_camera': {
        'onvif': camera,
        'move_request': {'ProfileToken': media_profile.token},
        'active': False
    }
}

# Create output folder for face bboxes
output_folder = "face_bboxes"
os.makedirs(output_folder, exist_ok=True)

def move_ptz(camera_name: str, pan, tilt, zoom):
    pan = max(-1, min(1, pan))
    tilt = max(-1, min(1, tilt))
    zoom = max(-1, min(1, zoom))
    
    try:
        onvif = cameras[camera_name]['onvif']
        move_request = cameras[camera_name]['move_request']
        onvif.get_service("ptz").ContinuousMove({
            'ProfileToken': move_request['ProfileToken'],
            'Velocity': {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': zoom}
            }
        })
        cameras[camera_name]['active'] = True
        logging.info(f"Moving PTZ: pan={pan:.4f}, tilt={tilt:.4f}, zoom={zoom:.4f}")
    except onvif.exceptions.ONVIFError as e:
        logging.error(f"ONVIF Error for camera {camera_name}: {e}")

def stop_ptz(camera_name: str) -> None:
    try:
        onvif = cameras[camera_name]['onvif']
        move_request = cameras[camera_name]['move_request']
        onvif.get_service("ptz").Stop(
            {
                "ProfileToken": move_request['ProfileToken'],
                "PanTilt": True,
                "Zoom": True,
            }
        )
        cameras[camera_name]['active'] = False
        logging.info(f"Stopped PTZ movement for camera: {camera_name}")
    except onvif.exceptions.ONVIFError as e:
        logging.error(f"ONVIF Error when stopping PTZ for camera {camera_name}: {e}")

frame_ready = threading.Event()
latest_frame = None
stop_thread = False

def read_frames():
    global latest_frame, stop_thread
    while not stop_thread:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
            frame_ready.set()
        else:
            logging.error("Failed to read frame")
            break

frame_thread = threading.Thread(target=read_frames)
frame_thread.start()

# Define center rectangle
center_rect_size = 200  # Size of the center rectangle
center_x, center_y = width // 2, height // 2
center_rect = (
    center_x - center_rect_size // 2,
    center_y - center_rect_size // 2,
    center_x + center_rect_size // 2,
    center_y + center_rect_size // 2
)

face_saved_ids = set()
state = "tracking"
zoom_step = 0.1
zoom_level = 1
last_id = None

try:
    while True:
        frame_ready.wait()
        frame = latest_frame.copy()
        frame_ready.clear()

        results = model.track(frame, show=False, classes=0, verbose=False, persist=True, tracker="bytetrack.yaml")[0]

        if results.boxes is not None and results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().tolist()
            min_track_id = min(track_ids)
            min_track_id_idx = track_ids.index(min_track_id)
            det = results.boxes[min_track_id_idx]
            xmin, ymin, xmax, ymax = map(int, det.xyxy[0])
            conf = det.conf[0]
            
            face_cx, face_cy = (xmin + xmax) // 2, (ymin + ymax) // 2

            # Calculate the area of intersection between face bbox and center rectangle
            intersection_area = max(0, min(xmax, center_rect[2]) - max(xmin, center_rect[0])) * \
                                max(0, min(ymax, center_rect[3]) - max(ymin, center_rect[1]))
            face_area = (xmax - xmin) * (ymax - ymin)
            intersection_ratio = intersection_area / face_area

            if state == "tracking":
                if intersection_ratio < 0.9: 
                    # Calculate pan and tilt to center the face
                    pan_velocity = -((face_cx / width) - 0.5) * 0.5
                    tilt_velocity = ((face_cy / height) - 0.5) * 0.5  # Changed sign here
                    zoom_velocity = 0

                    logging.info(f"Centering on face: pan={pan_velocity:.4f}, tilt={tilt_velocity:.4f}, zoom={zoom_velocity:.4f}")
                    move_ptz('main_camera', pan_velocity, tilt_velocity, zoom_velocity)
                else:
                    if min_track_id != last_id:
                        state = "zooming_in"
                        stop_ptz('main_camera')
                        face_saved_ids.discard(last_id)  # Allow new IDs to be processed
                        last_id = min_track_id

            elif state == "zooming_in":
                if zoom_level < 3:  # Assuming we want to zoom in up to 3x
                    zoom_velocity = zoom_step
                    move_ptz('main_camera', 0, 0, zoom_velocity)
                    zoom_level += zoom_step
                    logging.info(f"Zooming in: zoom_level={zoom_level:.2f}")
                else:
                    state = "snapshot"

            elif state == "snapshot":
                if min_track_id not in face_saved_ids:
                    face_img = frame[max(0, ymin):min(height, ymax), max(0, xmin):min(width, xmax)]
                    cv2.imwrite(os.path.join(output_folder, f"face_{min_track_id}_{time.time()}.jpg"), face_img)
                    logging.info(f"Saved face image for ID {min_track_id}")
                    face_saved_ids.add(min_track_id)
                state = "zooming_out"

            elif state == "zooming_out":
                if zoom_level > 1:
                    zoom_velocity = -zoom_step
                    move_ptz('main_camera', 0, 0, zoom_velocity)
                    zoom_level -= zoom_step
                    logging.info(f"Zooming out: zoom_level={zoom_level:.2f}")
                else:
                    state = "tracking"
                    stop_ptz('main_camera')

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'ID: {min_track_id}', (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'{conf:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            if state != "tracking":
                state = "tracking"
                last_id = None
                stop_ptz('main_camera')

        # Draw center rectangle
        cv2.rectangle(frame, (center_rect[0], center_rect[1]), (center_rect[2], center_rect[3]), (255, 0, 0), 2)

        cv2.imshow("Detections", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_thread = True
    frame_thread.join()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    stop_ptz('main_camera')