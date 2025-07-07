🚨 Restricted Area Face Tracking with PTZ Camera

This AI-powered surveillance system is designed to monitor individuals entering restricted or high-security areas using a PTZ-enabled ONVIF IP camera. It connects to the camera’s RTSP stream, detects faces in real time using YOLOv8, and tracks them with ByteTrack to maintain consistent identity. When a face is detected, the system automatically pans and tilts the camera to center the face in the frame, then zooms in smoothly. Once the face is properly zoomed and centered, a snapshot is captured and saved, and the system then zooms out to resume scanning for new faces. The full annotated video is recorded, and all cropped face images are stored separately for review or alerting purposes.

Working Pipeline:


Connect to RTSP stream from an ONVIF-enabled PTZ IP camera/n

Detect faces in real-time using YOLOv8 (yolov8n-face.pt)/n

Track face movement with ByteTrack to maintain ID

Automatically control PTZ:

  Pan/tilt to center the detected face

  Zoom in once face is centered

Capture and save a snapshot of the face

Zoom out and return to tracking

Save all cropped face images in face_bboxes/

Save the full output video with annotations as multnew.avi




    
