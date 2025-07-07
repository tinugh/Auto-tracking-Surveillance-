🚨 Restricted Area Face Tracking with PTZ Camera

This system is designed to monitor and track individuals entering a restricted area, such as a secured room or high-security zone. Using a PTZ-enabled IP camera, the system detects faces, automatically pans/tilts to center them, zooms in, captures a snapshot, and saves the full annotated video for record-keeping or alerting.


Working Pipeline

    Connect to RTSP Stream from an ONVIF-enabled PTZ IP camera.

    Detect Faces in Real-Time using YOLOv8 (yolov8n-face.pt).

    Track Face Movement using ByteTrack for ID persistence.

    Automatically Control PTZ:

        Pan and tilt the camera to center the face.

        Zoom in smoothly when the face is centered.

    Take Snapshot of the face and save it once zoomed.

    Zoom Out and return to tracking new faces.

    Save Output Video with bounding boxes and tracking info.

    All face crops saved in a dedicated folder.


    
