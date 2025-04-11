import streamlit as st
import PyPDF2
import io
import base64
import numpy as np
import tempfile
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Morse Code Translator",
    page_icon="📡",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title-container {
        background-color: #1e3a8a;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .content-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .output-container {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    .morse-text {
        font-family: monospace;
        font-size: 1.2rem;
        letter-spacing: 3px;
        line-height: 2;
        word-wrap: break-word;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #64748b;
    }
    .dot {
        color: #2563eb;
        font-weight: bold;
    }
    .dash {
        color: #9333ea;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Morse code dictionary
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--', 
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', ' ': '/', '.': '.-.-.-', ',': '--..--',
    '?': '..--..', "'": '.----.', '!': '-.-.--', '/': '-..-.', '(': '-.--.',
    ')': '-.--.-', '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-',
    '+': '.-.-.', '-': '-....-', '_': '..--.-', '"': '.-..-.', '$': '...-..-',
    '@': '.--.-.', '¿': '..-.-', '¡': '--...-'
}

# Add a reverse dictionary for decoding
REVERSE_MORSE_DICT = {value: key for key, value in MORSE_CODE_DICT.items()}

def text_to_morse(text):
    """Convert text to Morse code"""
    morse_code = []
    for char in text.upper():
        if char in MORSE_CODE_DICT:
            morse_code.append(MORSE_CODE_DICT[char])
        else:
            # For characters not in the dictionary, keep them as is
            morse_code.append(char)
    return ' '.join(morse_code)

def morse_to_text(morse):
    """Convert Morse code to text"""
    morse = morse.strip()
    morse_words = morse.split(' / ')
    text = []
    
    for word in morse_words:
        morse_chars = word.split(' ')
        for char in morse_chars:
            if char in REVERSE_MORSE_DICT:
                text.append(REVERSE_MORSE_DICT[char])
            elif char == '':
                continue
            else:
                text.append('?')  # Unknown Morse sequence
        text.append(' ')
    
    return ''.join(text).strip()

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_file(file):
    """Extract text from a general file"""
    text = file.getvalue().decode("utf-8")
    return text

def generate_morse_audio(morse_code, dot_duration=0.1):
    """Generate audio for Morse code using pure numpy arrays instead of pydub"""
    # Setup audio parameters
    sample_rate = 44100  # Hz
    amplitude = 0.5
    frequency = 800  # Hz
    
    # Durations in samples
    dot_samples = int(dot_duration * sample_rate)
    dash_samples = dot_samples * 3
    element_gap = dot_samples  # Gap between dots and dashes
    letter_gap = dot_samples * 3  # Gap between letters
    word_gap = dot_samples * 7  # Gap between words
    
    # Create an empty audio buffer
    audio_buffer = np.array([], dtype=np.float32)
    
    # Process each character in morse code
    for word in morse_code.split(' / '):
        for i, letter in enumerate(word.split(' ')):
            for j, symbol in enumerate(letter):
                if symbol == '.':
                    # Generate a dot
                    t = np.linspace(0, dot_duration, dot_samples, False)
                    tone = amplitude * np.sin(2 * np.pi * frequency * t)
                    audio_buffer = np.append(audio_buffer, tone)
                    
                elif symbol == '-':
                    # Generate a dash
                    t = np.linspace(0, dot_duration * 3, dash_samples, False)
                    tone = amplitude * np.sin(2 * np.pi * frequency * t)
                    audio_buffer = np.append(audio_buffer, tone)
                
                # Add gap between elements in the same letter
                if j < len(letter) - 1:
                    audio_buffer = np.append(audio_buffer, np.zeros(element_gap))
            
            # Add gap between letters in the same word
            if i < len(word.split(' ')) - 1:
                audio_buffer = np.append(audio_buffer, np.zeros(letter_gap))
        
        # Add gap between words
        audio_buffer = np.append(audio_buffer, np.zeros(word_gap))
    
    # Normalize audio to prevent clipping
    if np.max(np.abs(audio_buffer)) > 0:
        audio_buffer = audio_buffer / np.max(np.abs(audio_buffer))
    
    # Convert to 16-bit PCM
    audio_buffer_16bit = (audio_buffer * 32767).astype(np.int16)
    
    # Create a WAV file in memory
    import wave
    import struct
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        with wave.open(tmp_file.name, 'wb') as wav_file:
            # Set parameters
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes (16 bits)
            wav_file.setframerate(sample_rate)
            
            # Write frames
            for sample in audio_buffer_16bit:
                wav_file.writeframes(struct.pack('<h', sample))
        
        return tmp_file.name

def main():
    # Title
    st.markdown('<div class="title-container"><h1>📡 Morse Code Translator</h1></div>', unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Text Input", "File Upload", "Morse Decoder"])
    
    with tab1:
        st.subheader("Convert Text to Morse Code")
        text_input = st.text_area("Enter your text here:", height=150, 
                                placeholder="Type or paste your text here to convert to Morse code...")
        
        col1, col2 = st.columns(2)
        with col1:
            dot_duration = st.slider("Dot Duration (seconds)", 0.05, 0.5, 0.1, 0.05, 
                                    help="Adjust the speed of the Morse code audio. Lower values = faster code.")
        
        with col2:
            include_audio = st.checkbox("Generate Audio", value=True)
        
        if st.button("Convert to Morse", type="primary"):
            if text_input:
                morse_output = text_to_morse(text_input)
                
                st.markdown('<div class="output-container">', unsafe_allow_html=True)
                st.markdown(f"<div class='morse-text'>{visualize_morse(morse_output)}</div>", unsafe_allow_html=True)
                st.text_area("Morse Code (Text Format):", value=morse_output, height=100)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if include_audio:
                    try:
                        with st.spinner("Generating audio..."):
                            audio_file = generate_morse_audio(morse_output, dot_duration)
                            st.audio(audio_file, format='audio/wav')
                            
                            # Provide download button
                            with open(audio_file, "rb") as file:
                                btn = st.download_button(
                                    label="Download Audio",
                                    data=file,
                                    file_name="morse_audio.wav",
                                    mime="audio/wav"
                                )
                            
                            # Clean up temp file
                            try:
                                os.remove(audio_file)
                            except:
                                pass
                    except Exception as e:
                        st.error(f"Error generating audio: {str(e)}")
            else:
                st.warning("Please enter some text to convert.")
    
    with tab2:
        st.subheader("Convert File to Morse Code")
        
        file_type = st.radio("Select file type:", ("Text File (.txt)", "PDF File (.pdf)"))
        
        uploaded_file = st.file_uploader("Upload your file", 
                                        type=['txt', 'pdf'] if file_type == "PDF File (.pdf)" else ['txt'],
                                        help="Upload a text file or PDF to convert its content to Morse code.")
        
        col1, col2 = st.columns(2)
        with col1:
            dot_duration_file = st.slider("Dot Duration (seconds)", 0.05, 0.5, 0.1, 0.05, key="dot_file",
                                        help="Adjust the speed of the Morse code audio. Lower values = faster code.")
        
        with col2:
            include_audio_file = st.checkbox("Generate Audio", value=True, key="audio_file")
        
        if uploaded_file is not None:
            if st.button("Convert File to Morse", type="primary"):
                with st.spinner("Processing file..."):
                    if file_type == "PDF File (.pdf)":
                        extracted_text = extract_text_from_pdf(uploaded_file)
                    else:
                        extracted_text = extract_text_from_file(uploaded_file)
                    
                    st.markdown("### Extracted Text:")
                    st.text_area("", value=extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""), 
                                height=100, disabled=True)
                    
                    morse_output = text_to_morse(extracted_text)
                    
                    st.markdown('<div class="output-container">', unsafe_allow_html=True)
                    st.markdown("### Morse Code Output:")
                    st.markdown(f"<div class='morse-text'>{visualize_morse(morse_output[:500])}{'...' if len(morse_output) > 500 else ''}</div>", 
                                unsafe_allow_html=True)
                    st.text_area("Complete Morse Code:", value=morse_output, height=150)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if include_audio_file:
                        # For large files, limit the audio generation to the first 500 characters
                        morse_for_audio = morse_output[:500] if len(morse_output) > 500 else morse_output
                        try:
                            with st.spinner("Generating audio (this may take a moment)..."):
                                audio_file = generate_morse_audio(morse_for_audio, dot_duration_file)
                                st.audio(audio_file, format='audio/wav')
                                
                                # Provide download button
                                with open(audio_file, "rb") as file:
                                    btn = st.download_button(
                                        label="Download Audio",
                                        data=file,
                                        file_name="morse_audio.wav",
                                        mime="audio/wav",
                                        key="download_file_audio"
                                    )
                                
                                # Clean up temp file
                                try:
                                    os.remove(audio_file)
                                except:
                                    pass
                                
                            if len(morse_output) > 500:
                                st.info("Note: Audio has been generated for only the first portion of the text due to file size limitations.")
                        except Exception as e:
                            st.error(f"Error generating audio: {str(e)}")
    
    with tab3:
        st.subheader("Decode Morse Code to Text")
        morse_input = st.text_area("Enter Morse code here:", height=150,
                                placeholder="Enter Morse code here to decode (use dots and dashes, spaces between letters, and '/' between words)...\nExample: ... --- ...")
        
        if st.button("Decode Morse", type="primary"):
            if morse_input:
                try:
                    decoded_text = morse_to_text(morse_input)
                    
                    st.markdown('<div class="output-container">', unsafe_allow_html=True)
                    st.subheader("Decoded Text:")
                    st.write(decoded_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error decoding Morse code: {e}")
            else:
                st.warning("Please enter some Morse code to decode.")
    
    # End content container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions
    with st.expander("How to Use This Translator"):
        st.markdown("""
        ### Text to Morse
        1. Enter or paste text in the input area
        2. Adjust the dot duration slider to control the speed of the audio
        3. Click "Convert to Morse" to see and hear your text in Morse code
        
        ### File to Morse
        1. Select your file type (Text or PDF)
        2. Upload your file using the file uploader
        3. Click "Convert File to Morse" to process the file
        
        ### Morse to Text
        1. Enter Morse code in the input area using dots (.) and dashes (-)
        2. Use a single space between letters and a forward slash (/) between words
        3. Click "Decode Morse" to convert back to text
        
        ### Morse Code Format
        - Each letter is separated by a space
        - Each word is separated by a forward slash (/)
        - For example, "SOS" in Morse is "... --- ..."
        - "HELLO WORLD" is ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."
        """)
    
    # Morse code chart
    with st.expander("Morse Code Chart"):
        col1, col2, col3, col4 = st.columns(4)
        
        # Split the dictionary into chunks for display in columns
        items = list(MORSE_CODE_DICT.items())
        chunk_size = len(items) // 4
        
        for i, col in enumerate([col1, col2, col3, col4]):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 3 else len(items)
            
            for key, value in items[start_idx:end_idx]:
                if key == ' ':
                    key_display = '[space]'
                else:
                    key_display = key
                col.markdown(f"**{key_display}**: {visualize_morse(value)}", unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">© 2025 Morse Code Translator</div>', unsafe_allow_html=True)

def visualize_morse(morse_code):
    """Create a visual representation of the Morse code"""
    visual_morse = ""
    for char in morse_code:
        if char == '.':
            visual_morse += '<span class="dot">•</span>'
        elif char == '-':
            visual_morse += '<span class="dash">—</span>'
        elif char == ' ':
            visual_morse += '&nbsp;'
        elif char == '/':
            visual_morse += '&nbsp;/&nbsp;'
        else:
            visual_morse += char
    return visual_morse

if __name__ == "__main__":
    main()