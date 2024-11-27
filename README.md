import cv2

# بارگذاری فایل XML مربوط به تشخیص چهره
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# باز کردن دوربین یا خواندن فایل ویدئویی
cap = cv2.VideoCapture(0)  # در صورت تمایل می‌توانید مسیر فایل ویدئویی را به جای 0 قرار دهید
while True:
    # خواندن یک فریم از ویدئو
    ret, frame = cap.read()
    
    # تبدیل تصویر به مقیاس خاکستری
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # تشخیص چهره‌ها در تصویر
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # رسم مستطیل دور هر چهره
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # نمایش تصویر با مستطیل‌های رسم شده
    cv2.imshow('Face Detection', frame)
    
    # فشردن کلید 'q' برای خروج از حلقه
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# آزاد کردن منابع و بستن پنجره‌ها
cap.release()
cv2.destroyAllWindows()

