# parkinsons-voice-detection

A simple Streamlit app that predicts Parkinson's risk from a short voice recording.

## Features
- Record voice (say "aaaah" for 3–5 seconds)
- Audio preprocessing: mono conversion, resampling, normalization, silence trimming
- Feature extraction: pitch, jitter, shimmer, HNR/NHR
- Risk prediction: Low, Moderate, High (color-coded)

## Setup
1. Clone the repo:
```bash
git clone <repo-url>
cd <repo-folder>
```
2. Create and activate virtual environment
```bash
python -m venv venv
```
Windows
```bash
venv\Scripts\activate
```
macOS/Linux
```bash
source venv/bin/activate
```
3. Install dependencies
```bash
pip install requirements.txt
```
4. Run the app
```bash
streamlit run app.py
```
