import cv2

SCALE = 1.8
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
BANANA_PATH  = 'myaku.png'

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
banana_img   = cv2.imread(BANANA_PATH, cv2.IMREAD_UNCHANGED)
if banana_img is None or face_cascade.empty():
    raise FileNotFoundError("カスケードXMLまたは PNG が読み込めませんでした")

def overlay_image(bg_img, fg_img, x, y):
    """透過 PNG (fg_img) を背景 (bg_img) の (x,y) にアルファ合成"""
    fh, fw = fg_img.shape[:2]
    bh, bw = bg_img.shape[:2]

    if x < 0: fg_img = fg_img[:, -x:];      fw += x; x = 0
    if y < 0: fg_img = fg_img[-y:, :];      fh += y; y = 0
    if x+fw > bw: fg_img = fg_img[:, :bw-x]; fw = bw-x
    if y+fh > bh: fg_img = fg_img[:bh-y, :]; fh = bh-y
    if fh <= 0 or fw <= 0:
        return bg_img

    fg_rgb = fg_img[:, :, :3]
    alpha  = fg_img[:, :, 3] / 255.0
    for c in range(3):
        bg_img[y:y+fh, x:x+fw, c] = (1.0 - alpha) * bg_img[y:y+fh, x:x+fw, c] + alpha * fg_rgb[:, :, c]
    return bg_img

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラが開けませんでした")
    exit(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break
    if cv2.waitKey(1) != -1:
        break

    frame = cv2.flip(frame, 1)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        new_w = int(w * SCALE)
        new_h = int(h * SCALE)
        banana_resized = cv2.resize(banana_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        new_x = x - (new_w - w) // 2
        new_y = y - (new_h - h) // 2
        frame = overlay_image(frame, banana_resized, new_x, new_y)

    cv2.imshow('Myakuuu face', frame)

cap.release()
cv2.destroyAllWindows()
