import cv2
from datetime import datetime

WIN_NAME = 'Camera Preview'
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

cv2.namedWindow(WIN_NAME)

btn_w, btn_h = 150, 50
margin = 20
btn_x1, btn_y1 = width - btn_w - margin, height - btn_h - margin
btn_x2, btn_y2 = width - margin, height - margin

recording = False
writer = None
clicked = False

def on_mouse(event, x, y, flags, param):
    """Toggle recording if the button area is clicked."""
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            clicked = True

cv2.setMouseCallback(WIN_NAME, on_mouse)

def draw_button(frame, recording_flag):
    label = "STOP" if recording_flag else "REC"
    color = (0, 0, 255) if recording_flag else (50, 180, 50)
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), color, -1)
    cv2.putText(frame, label, (btn_x1 + 35, btn_y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if recording_flag:
        cv2.circle(frame, (20, 30), 8, (0, 0, 255), -1)  # red dot indicator

while True:
    ok, frame = cap.read()
    if not ok:
        break

    draw_button(frame, recording)
    cv2.imshow(WIN_NAME, frame)

    if recording and writer is not None:
        writer.write(frame)

    if clicked:
        clicked = False
        if not recording:
            filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use 'XVID' for .avi
            writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            recording = True
            print(f"Recording started → {filename}")
        else:
            recording = False
            if writer is not None:
                writer.release()
                writer = None
                print("Recording saved.")

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
