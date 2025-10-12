import cv2
import time
import datetime


# Configuration parameters
MAX_CAMERA_INDEX = 5
CAMERA_INIT_WAIT = 0.5
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
BODY_SCALE_FACTOR = 1.1
BODY_MIN_NEIGHBORS = 3
RECORD_FPS = 20.0
SECONDS_TO_RECORD_AFTER_DETECTION = 5
SHOW_FEED = True  # Set to False for faster performance on resource-constrained systems


def find_camera(max_index=MAX_CAMERA_INDEX, wait_sec=CAMERA_INIT_WAIT):
    """Try to open camera indices from 0..max_index and return an opened VideoCapture and index.
    Returns (cap, idx) or (None, None) if none found.
    """
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        # give the camera a tiny moment to initialize on some systems
        time.sleep(wait_sec)
        if cap.isOpened():
            return cap, i
        cap.release()
    return None, None


def load_cascades():
    """Load face and body cascade classifiers.
    Returns (face_cascade, body_cascade, success).
    """
    face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    body_xml = cv2.data.haarcascades + 'haarcascade_fullbody.xml'

    face_cascade = cv2.CascadeClassifier(face_xml)
    body_cascade = cv2.CascadeClassifier(body_xml)

    if face_cascade.empty():
        print(f"Failed to load face cascade at {face_xml}")
        return None, None, False
    
    if body_cascade.empty():
        print(f"Failed to load body cascade at {body_xml}")
        print("Body cascade not available; body detection will be disabled.")

    return face_cascade, body_cascade, True


def main():
    """Main security camera recording loop."""
    # Try to find a working camera device
    cap, cam_idx = find_camera()
    if cap is None:
        print("No camera found on indices 0..5. If you have a USB camera, try connecting it and rerun.")
        raise SystemExit(1)
    else:
        print(f"Opened camera index {cam_idx}")

    # Load cascade classifiers
    face_cascade, body_cascade, cascades_ok = load_cascades()
    if not cascades_ok:
        raise SystemExit(1)

    detection = False
    detection_stopped_time = None
    timer_started = False

    out = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # For tracking actual frame timing
    frame_count = 0
    start_time = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # camera disconnected or frame not ready
                print("Failed to grab frame from camera. Exiting loop.")
                break

            # to do the classification it needs grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # detects faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=FACE_SCALE_FACTOR, 
                                                   minNeighbors=FACE_MIN_NEIGHBORS)
            # detects bodies (if cascade loaded)
            bodies = []
            if not body_cascade.empty():
                bodies = body_cascade.detectMultiScale(gray, scaleFactor=BODY_SCALE_FACTOR, 
                                                       minNeighbors=BODY_MIN_NEIGHBORS)

            # IF we detect a body or a face, we'll start recording
            # this is our security camera logic!!
            if len(faces) + len(bodies) > 0:  # if we have at least one body or one face
                if detection:  # were we just detecting a body or face?
                    timer_started = False
                else:
                    detection = True  # we have detected something
                    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                    # determine frame size from actual frame
                    h, w = frame.shape[:2]
                    frame_size = (w, h)
                    
                    # Attempt to create video writer and verify it opened successfully
                    out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, RECORD_FPS, frame_size)
                    if out.isOpened():
                        frame_count = 0
                        start_time = time.time()
                        print(f"Started recording at {RECORD_FPS} FPS!")
                    else:
                        print(f"Warning: Failed to open video writer for {current_time}.mp4")
                        out = None
                        detection = False
                        
            elif detection:  # we lose detection of the face or body
                if timer_started:
                    if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                        detection = False
                        timer_started = False
                        if out is not None:
                            out.release()  # saves the video
                            out = None
                        print('Stop Recording!')
                else:
                    timer_started = True
                    detection_stopped_time = time.time()

            if detection and out is not None:
                out.write(frame)  # if we're recording then let's write to the frame
                frame_count += 1

            # optionally draw rectangles around detected faces/bodies
            # for (x, y, width, height) in faces:
            #     cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

            # Display camera feed if enabled
            if SHOW_FEED:
                cv2.imshow("Camera", frame)

            if cv2.waitKey(1) == ord('q'):
                break
                
    finally:
        if out is not None:
            out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
