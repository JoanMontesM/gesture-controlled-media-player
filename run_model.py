import tensorflow as tf
import numpy as np
import cv2
import pygame

model = tf.lite.Interpreter(model_path='hand_gesture_recognition.lite')
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

class_names = ['fist', 'five', 'okay', 'two']
conf_threshold = 0.5

class_colors = {
    'fist': (0, 0, 255),
    'five': (0, 255, 0),
    'okay': (255, 0, 0),
    'two': (0, 255, 255)
}

# Media Player logic
is_playing = False
volume = 50
song_idx = 0
playlist = ["music\Avanti - Time.mp3", "music\Aylex - Good Days.mp3", "music\Aylex - Where We Belong.mp3", "music\Burgundy - Mirrorball.mp3"] # songs paths

# gesture stabilization
last_gesture = None
same_gesture_count = 0
FRAMES_STABLE = 10


def load_play_song():
    global is_playing, volume, song_idx, playlist
    
    path = playlist[song_idx]
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(volume/100)
        pygame.mixer.music.play()
        is_playing = True
        print(f"[MEDIAPLAYER] Playing: {path}")
    except Exception as a:
        print("Error loading song")


def gesture_functionality(gesture):
    global is_playing, volume, song_idx
    
    if gesture == 'five':
        if is_playing:
            pygame.mixer.music.pause()
            is_playing = False
            state = "Pause"
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.unpause()
            else:
                load_play_song()
            is_playing = True
            state = "Play"
            
        print(f"[GESTURE] five -> Toggle play/pause -> {state}")
    
    elif gesture == 'okay':
        volume = min(100, volume + 5)
        pygame.mixer.music.set_volume(volume / 100)
        print(f"[GESTURE] okay -> Volume up -> {volume}")
    
    elif gesture == 'two':
        volume = max(0, volume - 5)
        pygame.mixer.music.set_volume(volume / 100)
        print(f"[GESTURE] two -> Volume down -> {volume}")
    
    elif gesture == 'fist':
        song_idx = (song_idx + 1) % len(playlist)
        print(f"[GESTURE] fist -> Next song -> {playlist[song_idx]}")
        load_play_song()


def process_frame(frame):
    
    # reshape to 320x320 (model input image size)
    frame_resized = cv2.resize(frame, (320, 320))

    # convert to RGB and normalize between 0 and 1
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image = img_rgb.astype(np.float32) / 255.0
    sample = np.expand_dims(image, axis=0)

    # inference
    model.set_tensor(input_details[0]['index'], sample)
    model.invoke()
    output = model.get_tensor(output_details[0]['index'])  # (1, 2100, 8)
    detections = output[0]  # (2100, 8)

    # computing the best detection and prepare the class ids
    class_scores = detections[:, 4:]
    confs = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])
    best_class_id = int(class_ids[best_idx])
    
    current_gesture = None

    # drawing a bounding box only if it's higher that the threshold
    if best_conf >= conf_threshold:
        # take x_min, x_max, y_min and y_max coordinates of the bbox
        x1_n, y1_n, x2_n, y2_n = detections[best_idx, :4]

        img_h, img_w = 320, 320

        x1 = int(x1_n * img_w)
        y1 = int(y1_n * img_h)
        x2 = int(x2_n * img_w)
        y2 = int(y2_n * img_h)

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        
        current_gesture = class_names[best_class_id]
        color = class_colors.get(current_gesture, (0, 255, 0))

        label = f"{current_gesture} ({best_conf:.2f})"

        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_resized, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
    
    return frame_resized, current_gesture


def drawing_ui(camera_frame, current_gesture):
    global is_playing, volume, song_idx, playlist
    
    UI_W, UI_H = 960, 540
    ui = np.zeros((UI_H, UI_W, 3), dtype=np.uint8)

    # background
    ui[:] = (200, 200, 200)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # project name
    cv2.putText(ui, "GESTURE MEDIA PLAYER", (40, 60), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    # track info
    cv2.putText(ui, "Use fist to change the song", (40, 100), font, 0.6, (0,0,0), 1, cv2.LINE_AA)
    track_text = f"Track path: {playlist[song_idx]}"
    cv2.putText(ui, track_text, (40, 130), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # play/pause
    cv2.putText(ui, "Use five to change between playing/paused", (40, 170), font, 0.6, (0,0,0), 1, cv2.LINE_AA)
    state_text = "PLAYING" if is_playing else "PAUSED"
    state_color = (57, 143, 0) if is_playing else (52, 50, 203)
    cv2.putText(ui, f"State: {state_text}", (40, 200), font, 0.6, state_color, 1, cv2.LINE_AA)

    # volume bar
    cv2.putText(ui, "Use okay/two to volume up/volume down", (40, 240), font, 0.6, (0,0,0), 1, cv2.LINE_AA)
    
    bar_x, bar_y = 40, 280
    bar_w, bar_h = 300, 25

    cv2.rectangle(ui, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 2)

    filled_w = int(bar_w * (volume / 100.0))
    cv2.rectangle(ui, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (100, 255, 100), -1)

    cv2.putText(ui, f"Volume: {volume}%", (bar_x, bar_y - 10), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # camera
    target_w, target_h = 320, 320

    cam_resized = cv2.resize(camera_frame, (target_w, target_h))

    margin = 20
    x0 = UI_W - target_w - margin
    y0 = margin

    ui[y0:y0 + target_h, x0:x0 + target_w] = cam_resized

    return ui


def main():
    global last_gesture, same_gesture_count
    
    pygame.mixer.init()
    
    cap = cv2.VideoCapture(0)
    
    print('Press "q" to leave')
    print('Gestures: five= play/pause, okay= volume up, two= volume down, fist= next song')
    
    # created to manage each frame, 
    # the action is only applied if the gesture is maintained during FRAMES_STABLE frames
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print('Some camera frame is not read correctly')
            break
        
        frame_resized, current_gesture = process_frame(frame)
        
        if current_gesture is None:
            same_gesture_count = 0
            last_gesture = None
        else:
            if current_gesture == last_gesture:
                same_gesture_count += 1
            else:
                last_gesture = current_gesture
                same_gesture_count += 1
            
            if same_gesture_count == FRAMES_STABLE:
                gesture_functionality(current_gesture)
                last_gesture = None
                same_gesture_count = 0
        
        ui_frame = drawing_ui(frame_resized, current_gesture)
        
        cv2.imshow("Gesture Media Player", ui_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()