import streamlit as st
from models.text2code1 import PseudoToCode
from models.code2text1 import CodeToPseudo

# Set page config with a bright/light theme look
st.set_page_config(
    page_title="Pseudocode to C++ Code and C++ Code to Pseudocode Generator",
    page_icon=":sparkles:",
    layout="wide",
    color="white",
)

# Inject custom CSS to enforce a bright theme
st.markdown(
    """
    <style>
    body {
        background-color:rgb(0, 0, 0);
        color: #000000;
    }
    .stApp {
        background-color:rgb(0, 0, 0);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Footer styling */
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        color: #666;
    }
    
    /* Creator profiles */
    .creator-profiles {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 10px;
    }
    .creator-profile {
        text-align: center;
    }
    .linkedin-button {
        display: inline-block;
        background-color: #0077B5;
        color: white !important;
        text-decoration: none;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-top: 5px;
        transition: background-color 0.3s;
    }
    .linkedin-button:hover {
        background-color: #005582;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use Streamlit's cache_resource for model initialization
@st.cache_resource
def load_pseudo_to_code_model():
    return PseudoToCode()

@st.cache_resource
def load_code_to_pseudo_model():
    return CodeToPseudo()

# Initialize models with caching
pseudo_to_code_model = load_pseudo_to_code_model()
code_to_pseudo_model = load_code_to_pseudo_model()

# Add session state to keep track of user inputs
if 'pseudocode_input' not in st.session_state:
    st.session_state.pseudocode_input = ""
if 'cpp_code_input' not in st.session_state:
    st.session_state.cpp_code_input = ""

# Streamlit application title
st.title("Pseudocode to C++ Code and C++ Code to Pseudocode Generator")

# Sidebar for user input
st.sidebar.header("User Input")

# Option to choose the conversion type
conversion_type = st.sidebar.selectbox("Select Conversion Type", 
                                       ("Pseudocode to C++ Code", "C++ Code to Pseudocode"))

# Input text area for pseudocode or C++ code
if conversion_type == "Pseudocode to C++ Code":
    pseudocode_input = st.text_area("Enter Pseudocode:", value=st.session_state.pseudocode_input)
    st.session_state.pseudocode_input = pseudocode_input
    
    if st.button("Generate C++ Code"):
        if pseudocode_input:
            with st.spinner("Generating C++ code..."):
                cpp_code = pseudo_to_code_model.generate_code(pseudocode_input)
            st.subheader("Generated C++ Code:")
            st.code(cpp_code, language="cpp")
        else:
            st.error("Please enter pseudocode to generate C++ code.")
else:
    cpp_code_input = st.text_area("Enter C++ Code:", value=st.session_state.cpp_code_input)
    st.session_state.cpp_code_input = cpp_code_input
    
    if st.button("Generate Pseudocode"):
        if cpp_code_input:
            with st.spinner("Generating pseudocode..."):
                pseudocode = code_to_pseudo_model.generate_pseudocode(cpp_code_input)
            st.subheader("Generated Pseudocode:")
            st.code(pseudocode)
        else:
            st.error("Please enter C++ code to generate pseudocode.")

