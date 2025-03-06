import os
from models.text2code.model1 import *
from models.text2code.model import Transformer
import pickle
import subprocess
import streamlit as st

class PseudoToCode:
    def __init__(self):
        self.MAX_LENGTH = 64
        self.D_MODEL = 512
        self.N_LAYERS = 4
        self.FFN_UNITS = 512
        self.N_HEADS = 8
        self.DROPOUT_RATE = 0.1

        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_weights_path = os.path.join(current_dir, 'text2code', 'text_to_code_transformer_weights.weights.h5')
        self.tokenizer_inputs_path = os.path.join(current_dir, 'text2code', 'tokenizer_inputs.pkl')
        self.tokenizer_outputs_path = os.path.join(current_dir, 'text2code', 'tokenizer_outputs.pkl')

        # Ensure Git LFS files are pulled
        if not os.path.exists(self.model_weights_path):
            try:
                subprocess.run(["git", "lfs", "pull"], check=True)
                self.input_tokenizer, self.output_tokenizer = self.load_tokenizer()
                self.model = self.load_model()
            except subprocess.CalledProcessError as e:
                st.error(f"Error pulling Git LFS files for PseudoCode to Code: {e}")
                self.input_tokenizer, self.output_tokenizer = None, None
                self.model = None
        else:
            self.input_tokenizer, self.output_tokenizer = self.load_tokenizer()
            self.model = self.load_model()

    def load_model(self):
        try:
            # Recalculate tokens
            num_words_inputs = self.input_tokenizer.vocab_size + 2
            num_words_output = self.output_tokenizer.vocab_size + 2

            transformer = Transformer(
                vocab_size_enc=num_words_inputs,
                vocab_size_dec=num_words_output,
                d_model=self.D_MODEL,
                n_layers=self.N_LAYERS,
                FFN_units=self.FFN_UNITS,
                n_heads=self.N_HEADS,
                dropout_rate=self.DROPOUT_RATE
            )

            # Dummy input to build the model
            dummy_enc_input = tf.ones((1, self.MAX_LENGTH), dtype=tf.int32)
            dummy_dec_input = tf.ones((1, self.MAX_LENGTH), dtype=tf.int32)
            transformer(dummy_enc_input, dummy_dec_input, training=False)

            # Load weights
            transformer.load_weights(self.model_weights_path)

            return transformer
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def load_tokenizer(self):
        try:
            # Load tokenizers
            with open(self.tokenizer_inputs_path, "rb") as f:
                tokenizer_inputs = pickle.load(f)

            with open(self.tokenizer_outputs_path, "rb") as f:
                tokenizer_outputs = pickle.load(f)
            return tokenizer_inputs, tokenizer_outputs
        except Exception as e:
            print(f"Error loading tokenizers: {e}")
            return None, None

    def generate_code(self, pseudocode):
        try:
            # Generate the C++ code
            predictions = translate(self.model, pseudocode, self.input_tokenizer, self.output_tokenizer, self.MAX_LENGTH)
            return predictions
        except Exception as e:
            print(f"Error generating code: {e}")
            return f"Error generating code: {e}"