# ⚡ Quick Start Guide

## **YES - Webcam Runs Locally! ✅**

All code executes **100% locally**. No cloud services. No external APIs. Complete privacy.

---

## 🚀 Launch Webcam Tracking (30 Seconds)

```bash
cd /workspaces/GAZE-
streamlit run webcam_ui.py
```

**Then:**
1. Open http://localhost:8502 in your browser
2. Click **"Webcam Tracking"** tab
3. Click **"▶️ Start Tracking"** button
4. Watch the red dot - your eyes follow it
5. View real-time gaze trajectory
6. Click **"⏹️ Stop"** when done
7. Click **"📊 Analyze Session"** for metrics

---

## 📊 What You Get

- **Real-time eye tracking** with facial landmarks
- **Eye region crops** showing pupil position
- **Gaze trajectory** visualization
- **ASD-like probability score** (0-100%)
- **Feature metrics**: fixation, saccades, entropy, etc.
- **All processed locally** on your machine

---

## 🧪 Run Examples (2 Minutes)

```bash
python example.py
```

Shows:
1. Face detection from webcam (10 frames)
2. Handcrafted feature extraction
3. Model training on synthetic data
4. End-to-end session analysis

---

## 🤖 Train Custom Model (5 Minutes)

```python
from src.train import DatasetManager, ModelTrainer

# Create dataset
dm = DatasetManager()
df = dm.create_synthetic_dataset(num_td=200, num_asd=200)
train, val, test = dm.split_train_val_test(df)

# Train
trainer = ModelTrainer("random_forest")
trainer.train_random_forest(
    train.drop("label", axis=1).values,
    train["label"].values,
    val.drop("label", axis=1).values,
    val["label"].values,
)

# Evaluate
metrics = trainer.evaluate(test.drop("label", axis=1).values, test["label"].values)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

---

## 📋 Dependencies Status

✅ PyTorch 2.9.1 (CPU)  
✅ OpenCV 4.12.0  
✅ MediaPipe 0.10.31  
✅ Streamlit 1.53.0  
✅ scikit-learn  
✅ pandas, numpy, scipy  

All installed and ready!

---

## 🎯 System Requirements

- **Python**: 3.8+ (you have 3.12+) ✅
- **RAM**: 2GB+ ✅
- **Storage**: ~500MB ✅
- **Webcam**: Optional (for live tracking) ✅
- **GPU**: Not required (CPU works fine) ✅

---

## 🔧 Troubleshooting

**"Webcam not detected?"**
→ Try demo mode in UI (synthetic data)

**"Module import error?"**
→ Run: `pip install -r requirements.txt`

**"Port 8502 already in use?"**
→ Run: `streamlit run webcam_ui.py --server.port 8503`

**"Need to train a model?"**
→ Example in README.md or run `python example.py`

---

## 📁 File Guide

| File | Purpose |
|------|---------|
| **webcam_ui.py** | Main Streamlit interface ⭐ START HERE |
| **example.py** | Working examples of all features |
| **config.py** | All settings & hyperparameters |
| **src/preprocessing.py** | Face detection & eye extraction |
| **src/feature_extraction.py** | 17 gaze metrics |
| **src/model.py** | ML models (RF + NN) |
| **src/train.py** | Training pipeline |

---

## ⚠️ Important Notes

✅ **Privacy**: Everything runs locally - no data leaves your computer  
✅ **Research Only**: Not a diagnostic tool - cannot diagnose autism  
✅ **Educational**: Great for learning gaze tracking & ML  
✅ **Open Source**: Full code available, modify as needed  

---

## Next Steps

1. **Run Streamlit**: `streamlit run webcam_ui.py`
2. **Explore Demo**: Try the demo mode with synthetic data
3. **Use Your Webcam**: Record your own tracking session
4. **Train a Model**: Follow example.py or README
5. **Review Code**: Check src/ modules for details

---

**Status**: Production-ready | All local | Privacy-first | Full documentation

Good luck! 🚀
