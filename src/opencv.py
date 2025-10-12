import cv2
import cv2
import time
import datetime


def find_camera(max_index=5, wait_sec=0.5):
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


# try to find a working camera device
cap, cam_idx = find_camera(max_index=5)
if cap is None:
    print("No camera found on indices 0..5. If you have a USB camera, try connecting it and rerun.")
    raise SystemExit(1)
else:
    print(f"Opened camera index {cam_idx}")


# classifier, there's a lot to pick from
# NOTE: correct filename is 'haarcascade_frontalface_default.xml' (not 'frontface')
face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
body_xml = cv2.data.haarcascades + 'haarcascade_fullbody.xml'

face_cascade = cv2.CascadeClassifier(face_xml)
body_cascade = cv2.CascadeClassifier(body_xml)

if face_cascade.empty():
    print(f"Failed to load face cascade at {face_xml}")
    raise SystemExit(1)
if body_cascade.empty():
    print(f"Failed to load body cascade at {body_xml}")
    # not fatal; continue but warn
    print("Body cascade not available; body detection will be disabled.")


detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

out = None
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

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
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # detects bodies (if cascade loaded)
        bodies = []
        if not body_cascade.empty():
            bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

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
                out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20.0, frame_size)
                print("Started recording!")
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
            out.write(frame) # if we're recording then let's write to the frame

        # optionally draw rectangles around detected faces/bodies
        # for (x, y, width, height) in faces:
        #     cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

        # title of the window that shows your camera feed
        cv2.imshow("Camera", frame) # shows your feed 

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # note: you can comment the top out to make things faster, especially
        # on a raspberry Pi, but then you won't see what the camera sees!!

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()

