# 🌦️ Rain Prediction App

Predict whether it will rain tomorrow in Australia using weather data and a trained neural network model.

![Streamlit App Screenshot](https://user-images.githubusercontent.com/yourusername/demo-screenshot.png) <!-- Optional: Add screenshot -->

---

## 📝 Description

**Rain Prediction App** is a web application built with Streamlit that utilizes a machine learning model to predict if it will rain the next day at a selected location in Australia. The prediction is based on historical and user-provided weather data. This app offers an intuitive UI, a dark-themed interface, and displays the model's confidence level in its predictions.

---

## 🚀 Live Demo

👉 [Launch the App on Streamlit Cloud](https://aus-rain-prediction-app.streamlit.app)

---

## 🔧 Features

- 🧠 Trained neural network (Keras) for binary classification (`RainTomorrow`: Yes/No)
- 📍 One-hot encoded locations across 49 Australian cities
- 🌬️ Wind direction encoded using sine/cosine transformations
- 📆 Temporal features encoded cyclically (day/month)
- ⚖️ Inputs normalized using a pre-fitted StandardScaler
- 🌙 Dark theme for an elegant look
- 📈 Model confidence score displayed alongside prediction

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Model**: Keras (TensorFlow backend)
- **Preprocessing**: Scikit-learn, Pandas, NumPy
- **Deployment**: Streamlit Community Cloud

---

## 📥 Inputs

The following features must be entered in the sidebar:

- 📅 Date
- 📍 Location (dropdown from 49 options)
- 🌡️ Min Temperature (°C)
- 🌧️ Rainfall (mm)
- 🧭 Gust Direction, Wind Direction at 9AM & 3PM
- 💨 Wind Gust Speed (km/h), Wind Speed at 3PM (km/h)
- 💧 Humidity at 9AM & 3PM (%)
- 🌬️ Pressure at 9AM (hPa)
- ☁️ Cloud Cover at 3PM (oktas)
- ☀️ Rain Today? (Yes/No)

---

## 📤 Output

- **Rain Tomorrow?** — Displayed as: ☀️ No / 🌧️ Yes  
- **Confidence** — Model’s certainty about the prediction, displayed as a percentage.

---

## 📂 Project Structure



rain-prediction-app/
├── Rain\_predict.py             # Main Streamlit app
├── model.h5                    # Trained neural network model
├── scaler.joblib               # Scaler for feature normalization
├── requirements.txt            # Dependencies
└── .streamlit/
└── config.toml             # Streamlit theme configuration



---

## ⚙️ Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/rain-prediction-app.git
cd rain-prediction-app
```

2. **Create a Virtual Environment & Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the App**

```bash
streamlit run Rain_predict.py
```

---

## 🧠 Model Info

The neural network model was trained on the "Rain in Australia" dataset, with preprocessing steps including:

* Cyclic encoding of date features
* Directional encoding with sine/cosine
* One-hot encoding of categorical data
* Normalization using StandardScaler
* Training with SGD optimizer, ReLU activations, and cross-entropy loss

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [Rain in Australia Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) – from Kaggle
* [Streamlit](https://streamlit.io/) – for app framework
* [TensorFlow / Keras](https://www.tensorflow.org/) – for model training

---

## ✨ Author

Developed with ❤️ by [Abhinav Raj](https://github.com/ARJ010)



