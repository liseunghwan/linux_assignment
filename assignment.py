import cv2
import time
import RPi.GPIO as GPIO
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import threading
import asyncio

# === 설정 ===
TOKEN = 'YOUR_BOT_TOKEN'  # ← 본인의 봇 토큰으로 교체
BUTTON_PIN = 21
LED_PIN = 6

# === 전역 변수 ===
app = None
loop = None
detect_on = False
current_chat_id = None

# === Haar 모델 경로 (상대경로) ===
face_path = './haar-cascade-files-master/haarcascade_frontalface_default.xml'
eye_path = './haar-cascade-files-master/haarcascade_eye.xml'
face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)

# === GPIO 설정 ===
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUTTON_PIN, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(LED_PIN, GPIO.OUT)

# === 버튼 콜백 ===
def button_callback(pin):
    global detect_on
    detect_on = not detect_on
    status = "ON" if detect_on else "OFF"
    print(f"[BUTTON] 얼굴 인식 상태: {status}")

    if current_chat_id and app and loop:
        loop.call_soon_threadsafe(lambda: asyncio.create_task(
            app.bot.send_message(chat_id=current_chat_id, text=f"얼굴 인식이 {status} 상태로 전환되었습니다."))
        )

GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=button_callback, bouncetime=200)

# === /start 명령 처리 ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_chat_id
    current_chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"얼굴 인식 봇에 오신 것을 환영합니다.\n"
        f"이 채팅의 chat_id는 {current_chat_id} 입니다.\n"
        "버튼을 눌러 얼굴 인식을 켜고 끌 수 있어요.",
        parse_mode=None
    )
    print(f"[INFO] chat_id 등록됨: {current_chat_id}")

# === 얼굴 + 눈 감지 루프 ===
def face_eye_detect_loop():
    global loop
    while True:
        if detect_on and current_chat_id:
            camera = cv2.VideoCapture(0, cv2.CAP_V4L)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera.grab()
            ret, image = camera.read()

            if not ret:
                camera.release()
                time.sleep(1)
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_roi_gray = gray[y:y+h, x:x+w]
                face_roi_color = image[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_roi_gray)

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

                if len(eyes) == 2:
                    filename = f"/tmp/face_eye_{int(time.time())}.jpg"
                    cv2.imwrite(filename, image)

                    GPIO.output(LED_PIN, GPIO.HIGH)
                    print("[DETECT] 얼굴 + 눈 2개 감지됨 → 사진 전송 + LED ON")

                    if loop and app:
                        loop.call_soon_threadsafe(lambda: asyncio.create_task(
                            app.bot.send_photo(chat_id=current_chat_id, photo=open(filename, 'rb'))
                        ))

                    time.sleep(3)
                    GPIO.output(LED_PIN, GPIO.LOW)
                    time.sleep(10)
                    break

            # 화면 출력
            cv2.imshow('Face Detection Preview', image)
            cv2.waitKey(1)

            camera.release()
        time.sleep(1)

# === 메인 ===
if __name__ == '__main__':
    try:
        print("얼굴+눈 감지 봇 실행 중... /start 입력 후 버튼으로 제어하세요.")
        app = ApplicationBuilder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))

        # 루프 가져오기
        loop = asyncio.get_event_loop()

        # 얼굴 인식 루프 실행
        threading.Thread(target=face_eye_detect_loop, daemon=True).start()

        # 텔레그램 봇 실행
        app.run_polling()

    except KeyboardInterrupt:
        print("\n프로그램 종료 중...")

    finally:
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("GPIO 정리 완료")
