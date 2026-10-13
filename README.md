# BinBot

## How to run it

#### 1. Clone the project

```bash
git clone https://github.com/Liamrillah/BinBot.git
cd BinBot
```

#### 2. Set up the environment

```bash
python -m venv .venv
```

#### 3. Activate the environment

For Windows CMD:

```bat
.venv\\Scripts\\activate
```

For PowerShell:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

#### 4. Install dependencies

```bash
pip install -r requirements.txt
```

#### 5. Collect the dataset

Run `capture_dataset.py` to collect hand gesture images using your webcam.

```bash
python capture_dataset.py
```

Alternatively, you can download a suitable hand gesture dataset from Kaggle or another source. Organize the images into classes such as `Go` and `Stop`.

#### 6. Train the model

Open [Google Teachable Machine](https://teachablemachine.withgoogle.com/) and follow these steps:

1. Create a new **Image Project**.
2. Create a class for each gesture, such as `Go` and `Stop`.
3. Upload the corresponding dataset images to each class.
4. Train the model.
5. Export the trained model in **TensorFlow/Keras format**.

#### 7. Add the trained model

Place the exported model files in your project directory. Depending on the export format, these may include `keras_model.h5` and `labels.txt`.

Update the model paths in the prediction script to match the location of the exported files.

#### 8. Test hand detection

Run `test_gesture.py` to test hand detection and image preprocessing with your webcam.

```bash
python test_gesture.py
```

**Note:** This `test_gesture.py` script only tests hand detection and preprocessing; it does not run predictions using the trained Teachable Machine model. To test the trained model, run the model prediction script included in your project.
"""
