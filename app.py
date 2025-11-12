import streamlit as st
import io
import soundfile as sf
import numpy as np
from src.preprocessing import preprocess_audio
from src.feature_extraction import extract_features
from src.predict_risk import predict

st.set_page_config(page_title="Parkinson's Risk Prediction")
st.title("Parkinson's Voice Screening")
st.write("### Step 1: Record your voice")
st.write("Say 'aaaah' steadily for 3-5 seconds")
st.write("Repeat 3-4 times")
st.info("Tip: If recording stops early, try varying your volume slightly")

record_audio = st.audio_input("Record a voice message, testing!")

if record_audio:
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(record_audio.read()))

        st.success("Recording received!")
        duration = len(audio_data)/sample_rate
        
        st.write(f"Sample rate: {sample_rate} Hz")
        st.write(f"Duration: {duration:.2f} seconds")
        st.write(f"Shape: {audio_data.shape}")
        
        if duration < 3:
            st.warning("Too short recording, need atleast 3 seconds, please :)")
        elif duration > 10:
            st.warning("Too long recording, keep it under 10 seconds, please :)")
        else:
            st.success("Duration is good! yay :3")

            if st.button("Analyze recording and Predict Risk", type="primary"):
                st.write("### Step 2: Preprocessing audio")
                with st.spinner("Cleaning audio..."):
                    clean_audio, clean_sr = preprocess_audio(audio_data, sample_rate)
                    clean_duration = len(clean_audio)/clean_sr

                st.success("Audio cleaned successfully!")
                st.write(f"Original duration: {duration:.2f}")
                st.write(f"Cleaned duration: {clean_duration:.2f}")

                if clean_duration < 1:
                    st.error("Audio too short, try recording longer, louder")
                else:
                    with st.spinner("Analyzing vocal features..."):
                        features = extract_features(clean_audio, clean_sr)

                    if features is None:
                        st.error("Feature extraction failed")
                    else:
                        st.success(f"Extracted {len(features)} features!")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("Pitch features: ")
                            st.metric("Average f0", f"{features.get('MDVP:Fo(Hz)', 0):.1f} Hz")
                            st.metric("Max f0", f"{features.get('MDVP:Fhi(Hz)', 0):.1f} Hz")
                            st.metric("Min f0", f"{features.get('MDVP:Flo(Hz)', 0):.1f} Hz")

                            st.write("Jitter features: ")
                            st.metric("Local jitter", f"{features.get('MDVP:Jitter(%)', 0):.5f}")
                            st.metric("Absolute jitter", f"{features.get('MDVP:Jitter(Abs)', 0):.6f}")
                            st.metric("RAP", f"{features.get('MDVP:RAP', 0):.6f}")
                            st.metric("PPQ", f"{features.get('MDVP:PPQ', 0):.6f}")
                            st.metric("DDP", f"{features.get('Jitter:DDP', 0):.6f}")

                        with col2:
                            st.write("Shimmer features: ")
                            st.metric("Local shimmer", f"{features.get('MDVP:Shimmer', 0):.5f}")
                            st.metric("Shimmer(dB)", f"{features.get('MDVP:Shimmer(dB)', 0):.3f}")
                            st.metric("APQ3", f"{features.get('Shimmer:APQ3', 0):.5f}")
                            st.metric("APQ5", f"{features.get('Shimmer:APQ5', 0):.5f}")
                            st.metric("APQ11", f"{features.get('MDVP:APQ', 0):.5f}")
                            st.metric("DDA", f"{features.get('Shimmer:DDA', 0):.5f}")

                            st.write("Voice quality")
                            st.metric("HNR", f"{features.get('HNR', 0):.2f} dB")
                            st.metric("NHR", f"{features.get('NHR', 0):.5f}")

                        with st.spinner("Predicting Parkinson's risk..."):
                            risk = predict(features)
                            risk_score = risk["probability"]

                        st.success(f"Predicted Parkinson's risk: {risk_score*100}%")
                        if risk_score > 0.7:
                            st.warning("High likelihood of Parkinson's symptoms, Consult with a doctor")
                        elif risk_score<0.7 and risk_score>0.4:
                            st.info("Moderate likelihood of Parkinson's symptoms, Keep monitoring and consult in unsure")
                        else:
                            st.success("Low likelihood of Parkinson's symptoms, Consult a doctor if unsure")

    except Exception as e:
        st.error(f"Error occurred while processing: {e}")
        st.write("Try recording again or check the microphone")