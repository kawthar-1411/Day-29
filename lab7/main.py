from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# تحميل النموذج والمقياس
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# إنشاء تطبيق FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to Player Price Prediction API"}

# تعريف Pydantic model للتحقق من صحة البيانات المدخلة
class InputFeatures(BaseModel):
    appearance: int
    minutes_played: int
    award: int
    highest_value: int

# معالجة البيانات المدخلة
def preprocessing(input_features: InputFeatures):
    # تحويل المدخلات إلى مصفوفة numpy (التي يمكن تمريرها إلى النموذج)
    input_data = np.array([[input_features.appearance,
                            input_features.minutes_played,
                            input_features.award,
                            input_features.highest_value]])

    # استخدام المقياس لتحويل البيانات (تقييس القيم)
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

# تحديد المسار للتنبؤ
@app.get("/predict")
def predict(input_features: InputFeatures):
    print(f"Received input: {input_features}")  # طباعة المدخلات للتأكد
    processed_data = preprocessing(input_features)
    prediction = model.predict(processed_data)
    print(f"Prediction result: {prediction}")  # طباعة النتيجة
    return {"prediction": prediction[0]}  # إرجاع التنبؤ كـ JSON
