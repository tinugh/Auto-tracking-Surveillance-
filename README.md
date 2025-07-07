🚨 Restricted Area Face Tracking with PTZ Camera

This AI-powered surveillance system is designed to monitor individuals entering restricted or high-security areas using a PTZ-enabled ONVIF IP camera. It connects to the camera’s RTSP stream and uses YOLOv8 to perform real-time face detection. Face movement is tracked using ByteTrack, which ensures consistent ID assignment for each individual. Once a face is detected, the system automatically pans and tilts the camera to center the subject within the frame. When the face is aligned in the center, it smoothly zooms in, captures a snapshot, and saves the cropped face image to a dedicated folder. After saving the snapshot, the camera zooms out to resume scanning for new individuals. Throughout the process, the system continuously records and saves an annotated output video showing bounding boxes, tracking IDs, and camera movement in action. All face crops are saved in the directory.




    
