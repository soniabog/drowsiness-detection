import os
import cv2
import dlib
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

input_folder = 'example_in'
output_folder = 'example_out'
os.makedirs(output_folder, exist_ok=True)

def largest_face(rects):
    return max(rects, key=lambda rect: rect.width() * rect.height()) if rects else None

def add_margin_to_eye(x, y, w, h, width, height, margin_w=0.6, upper_margin_h=2.5, lower_margin_h=0.7):
    x_margin = int(w * margin_w)
    upper_y_margin = int(h * upper_margin_h)
    lower_y_margin = int(h * lower_margin_h)
    x_new = max(x - x_margin, 0)
    y_new = max(y - upper_y_margin, 0)
    x_end = min(x + w + x_margin, width)
    y_end = min(y + h + lower_y_margin, height)
    return (x_new, y_new, x_end - x_new, y_end - y_new)

def adjust_and_label_eye(image, roi, eye_label, window_name="Adjust ROI"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    orig_x1, orig_y1, orig_x2, orig_y2 = roi

    def nothing(x):
        pass

    cv2.createTrackbar("Left", window_name, 0, 100, nothing)
    cv2.createTrackbar("Right", window_name, 0, 100, nothing)
    cv2.createTrackbar("Up", window_name, 0, 100, nothing)
    cv2.createTrackbar("Down", window_name, 0, 100, nothing)

    while True:
        left = cv2.getTrackbarPos("Left", window_name)
        right = cv2.getTrackbarPos("Right", window_name)
        up = cv2.getTrackbarPos("Up", window_name)
        down = cv2.getTrackbarPos("Down", window_name)

        x1_new = max(orig_x1 - left, 0)
        y1_new = max(orig_y1 - up, 0)
        x2_new = min(orig_x2 + right, image.shape[1])
        y2_new = min(orig_y2 + down, image.shape[0])

        roi_img = image[y1_new:y2_new, x1_new:x2_new].copy()
        cv2.imshow(window_name, roi_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    while True:
        cv2.imshow(window_name, roi_img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("o"):
            eye_class = "eye_open"
            break
        elif key == ord("c"):
            eye_class = "eye_closed"
            break

    cv2.destroyWindow(window_name)
    return [x1_new, y1_new, x2_new, y2_new, eye_class]

class_dict = {"eye_open": 0, "eye_closed": 1, "face": 2}

for img_name in os.listdir(input_folder):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(input_folder, img_name)
    image = cv2.imread(image_path)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    face = largest_face(detector(gray, 1))

    if face is None:
        print(f"No face in {img_name}")
        continue

    shape = face_utils.shape_to_np(predictor(gray, face))

    (x, y, fw, fh) = face.left(), face.top(), face.width(), face.height()
    x1_face, y1_face = max(x, 0), max(y, 0)
    x2_face, y2_face = min(x + fw, w), min(y + fh, h)

    (x, y, w_eye, h_eye) = cv2.boundingRect(np.array([shape[face_utils.FACIAL_LANDMARKS_IDXS['left_eye'][0]:face_utils.FACIAL_LANDMARKS_IDXS['left_eye'][1]]]))
    left_eye_roi = add_margin_to_eye(x, y, w_eye, h_eye, image.shape[1], image.shape[0])
    left_eye_roi = [left_eye_roi[0], left_eye_roi[1], left_eye_roi[0]+left_eye_roi[2], left_eye_roi[1]+left_eye_roi[3]]

    (x, y, w_eye, h_eye) = cv2.boundingRect(np.array([shape[face_utils.FACIAL_LANDMARKS_IDXS['right_eye'][0]:face_utils.FACIAL_LANDMARKS_IDXS['right_eye'][1]]]))
    right_eye_roi = add_margin_to_eye(x, y, w_eye, h_eye, image.shape[1], image.shape[0])
    right_eye_roi = [right_eye_roi[0], right_eye_roi[1], right_eye_roi[0]+right_eye_roi[2], right_eye_roi[1]+right_eye_roi[3]]

    adjusted_left_eye = adjust_and_label_eye(image, left_eye_roi, "Left Eye", "LEFT EYE ADJUST")
    adjusted_right_eye = adjust_and_label_eye(image, right_eye_roi, "Right Eye", "RIGHT EYE ADJUST")

    final_img = image.copy()
    for adjusted_eye in [adjusted_left_eye, adjusted_right_eye]:
        color = (0,255,0) if adjusted_eye[4] == 'eye_open' else (0,0,255)
        cv2.rectangle(final_img, (adjusted_eye[0], adjusted_eye[1]), (adjusted_eye[2], adjusted_eye[3]), color, 2)
        cv2.putText(final_img, adjusted_eye[4], (adjusted_eye[0], adjusted_eye[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("FINAL LABELED RESULT (press any key)", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    txt_filename = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(output_folder, txt_filename)

    with open(txt_path, "w") as f:
        for roi in [adjusted_left_eye, adjusted_right_eye]:
            x1, y1, x2, y2, label = roi
            xc, yc, rw, rh = (x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h
            class_id = class_dict[label]
            f.write(f"{class_id} {xc:.4f} {yc:.4f} {rw:.4f} {rh:.4f}\n")

    with open(txt_path, "r") as f:
        labels = f.readlines()
        if any(line.startswith("2 ") for line in labels):
            print(f"Face labeled in: {img_name}")
            continue

    xc = (x1_face + x2_face) / 2 / w
    yc = (y1_face + y2_face) / 2 / h
    bw = (x2_face - x1_face) / w
    bh = (y2_face - y1_face) / h

    with open(txt_path, "a") as f:
        f.write(f"2 {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}\n")

    print(f"Labels saved: {txt_filename}")
