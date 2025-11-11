import streamlit as st
import io
import soundfile as sf
import numpy as np

st.title("Parkinson's Voice Screening")
st.write("### Step 1: Record your voice")
st.write("Say 'aaaah' steadily for 3-5 seconds")
st.info("💡 Tip: If recording stops early, try varying your volume slightly")

record_audio = st.audio_input("Record a voice message, testing!")

if record_audio:
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(record_audio.read()))

        st.success("Recording received!")
        duration = len(audio_data)/sample_rate

        if duration < 3:
            st.warning("Too short recording, need atleast 3 seconds, please :)")
        elif duration > 10:
            st.warning("Too long recording, keep it under 10 seconds, please :)")
        else:
            st.success("Duration is good! yay :3")

        st.write(f"Sample rate: {sample_rate} Hz")
        st.write(f"Duration: {duration:.2f} seconds")
        st.write(f"Shape: {audio_data.shape}")

        # st.audio(record_audio)

    except Exception as e:
        st.error(f"Error occurred while processing: {e}")