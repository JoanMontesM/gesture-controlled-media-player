import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import pygame

# ---------------------------
# Configuración modelo TFLite
# ---------------------------

MODEL_PATH = 'hand_gesture_recognition.lite'
CLASS_NAMES = ['fist', 'five', 'okay', 'two']  # ajusta si tu modelo usa otro orden
CONF_THRESHOLD = 0.5

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Colores por gesto (BGR)
GESTURE_COLORS = {
    'fist': (0, 0, 255),    # rojo
    'five': (0, 255, 0),    # verde
    'okay': (255, 0, 0),    # azul
    'two': (0, 255, 255),   # amarillo
}

# ---------------------------
# Reproductor de música
# ---------------------------

class GestureMusicPlayer:
    def __init__(self, music_folder='music'):
        pygame.mixer.init()
        self.music_folder = music_folder
        self.playlist = self._load_playlist(music_folder)
        self.current_index = 0
        self.volume = 0.5  # [0.0, 1.0]
        pygame.mixer.music.set_volume(self.volume)
        self.is_paused = False

        if not self.playlist:
            print(f"[AVISO] No se encontraron archivos de audio en '{music_folder}'")
        else:
            print("[INFO] Playlist cargada:")
            for i, track in enumerate(self.playlist):
                print(f"  {i}: {os.path.basename(track)}")
            self._load_current_track()

    def _load_playlist(self, folder):
        exts = ('*.mp3', '*.wav', '*.ogg')
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(files)

    def _load_current_track(self):
        if not self.playlist:
            return
        track_path = self.playlist[self.current_index]
        pygame.mixer.music.load(track_path)
        print(f"[TRACK] Cargado: {os.path.basename(track_path)}")

    def get_current_track_name(self):
        if not self.playlist:
            return "No hay pistas"
        return os.path.basename(self.playlist[self.current_index])

    def play_pause(self):
        if not self.playlist:
            print("[PLAY/PAUSE] No hay pistas en la playlist.")
            return

        if pygame.mixer.music.get_busy() and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            print("[MÚSICA] Pausa")
        elif self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            print("[MÚSICA] Reanudar")
        else:
            self._load_current_track()
            pygame.mixer.music.play()
            self.is_paused = False
            print("[MÚSICA] Play")

    def stop(self):
        pygame.mixer.music.stop()
        self.is_paused = False
        print("[MÚSICA] Stop")

    def next_track(self):
        if not self.playlist:
            print("[NEXT] No hay pistas.")
            return
        self.current_index = (self.current_index + 1) % len(self.playlist)
        self._load_current_track()
        pygame.mixer.music.play()
        self.is_paused = False
        print("[MÚSICA] Siguiente pista")

    def volume_up(self, step=0.1):
        self.volume = min(1.0, self.volume + step)
        pygame.mixer.music.set_volume(self.volume)
        print(f"[VOLUMEN] + → {self.volume:.2f}")

    def volume_down(self, step=0.1):
        self.volume = max(0.0, self.volume - step)
        pygame.mixer.music.set_volume(self.volume)
        print(f"[VOLUMEN] - → {self.volume:.2f}")

    def get_volume_percent(self):
        return int(self.volume * 100)

    def get_status(self):
        if pygame.mixer.music.get_busy():
            if self.is_paused:
                return "Pausado"
            else:
                return "Reproduciendo"
        else:
            return "Detenido"

# ---------------------------
# Lógica principal
# ---------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    print("Gestos (reproductor):")
    print("  fist  -> siguiente canción")
    print("  five  -> play / pause")
    print("  okay  -> volumen +")
    print("  two   -> volumen -")
    print("Pulsa 'q' para salir.")

    player = GestureMusicPlayer(music_folder='music')

    # Gesto que ya ha disparado acción.
    # La acción se lanza solo cuando el gesto CAMBIA.
    last_action_gesture = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame de la cámara")
            break

        # -------- Cámara + modelo --------
        frame_resized = cv2.resize(frame, (320, 320))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image = img_rgb.astype(np.float32) / 255.0
        sample = np.expand_dims(image, axis=0)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])  # (1, 2100, 8)
        detections = output[0]

        class_scores = detections[:, 4:]
        confs = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])
        best_class_id = int(class_ids[best_idx])

        current_gesture_name = None

        if best_conf >= CONF_THRESHOLD:
            # coords normalizadas
            x1_n, y1_n, x2_n, y2_n = detections[best_idx, :4]
            img_h, img_w = 320, 320

            x1 = int(x1_n * img_w)
            y1 = int(y1_n * img_h)
            x2 = int(x2_n * img_w)
            y2 = int(y2_n * img_h)

            # recorte seguro
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            current_gesture_name = CLASS_NAMES[best_class_id]
            color = GESTURE_COLORS.get(current_gesture_name, (0, 255, 0))

            # bounding box + etiqueta en la VENTANA DE CÁMARA
            label = f"{current_gesture_name} ({best_conf:.2f})"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            current_gesture_name = None

        # -------- Mapeo gesto -> acción (SOLO al cambiar de gesto) --------
        if current_gesture_name is not None:
            if current_gesture_name != last_action_gesture:
                if current_gesture_name == 'fist':
                    player.next_track()
                elif current_gesture_name == 'five':
                    player.play_pause()
                elif current_gesture_name == 'okay':
                    player.volume_up()
                elif current_gesture_name == 'two':
                    player.volume_down()

                # este gesto ya ha ejecutado su acción
                last_action_gesture = current_gesture_name
        else:
            # si no hay gesto, "liberamos" para que el siguiente gesto cuente como nuevo
            last_action_gesture = None

        # -------- Interfaz del reproductor --------
        ui_h, ui_w = 250, 600
        ui_frame = np.zeros((ui_h, ui_w, 3), dtype=np.uint8)

        # Título
        cv2.putText(ui_frame, "Gesture Music Player", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Info de pista
        track_name = player.get_current_track_name()
        cv2.putText(ui_frame, f"Pista: {track_name}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Estado
        status = player.get_status()
        cv2.putText(ui_frame, f"Estado: {status}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        # Volumen
        vol = player.get_volume_percent()
        cv2.putText(ui_frame, f"Volumen: {vol}%", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Controles (texto estático de ayuda)
        controls_text = [
            "Gestos:",
            "fist  -> siguiente pista",
            "five  -> play / pause",
            "okay  -> volumen +",
            "two   -> volumen -",
        ]
        y0 = 60
        for i, txt in enumerate(controls_text):
            cv2.putText(ui_frame, txt, (320, y0 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Mostrar ventanas
        cv2.imshow("Camara - Gestos", frame_resized)
        cv2.imshow("Reproductor", ui_frame)

        # salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()