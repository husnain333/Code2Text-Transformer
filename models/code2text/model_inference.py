import tensorflow as tf

def predict(transformer, inp_sentence, tokenizer_in, tokenizer_out, sos_token_input, eos_token_input, sos_token_output, eos_token_output, target_max_len):
    # Tokenize the input sequence using the tokenizer_in
    inp_sentence = sos_token_input + tokenizer_in.encode(inp_sentence) + eos_token_input
    enc_input = tf.expand_dims(inp_sentence, axis=0)

    # Set the initial output sentence to sos
    out_sentence = sos_token_output
    # Reshape the output
    output = tf.expand_dims(out_sentence, axis=0)

    # For max target len tokens
    for _ in range(target_max_len):
        # Call the transformer and get the logits 
        predictions = transformer(enc_input, output, training=False) #(1, seq_length, VOCAB_SIZE_ES)
        # Extract the logists of the next word
        prediction = predictions[:, -1:, :]
        # The highest probability is taken
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        # Check if it is the eos token
        if predicted_id == eos_token_output:
            return tf.squeeze(output, axis=0)
        # Concat the predicted word to the output sequence
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

# Update the translate function to use the model and tokens provided
def translate(model, sentence, tokenizer_in, tokenizer_out, MAX_LENGTH, device=None):
    # Recalculate tokens
    num_words_inputs = tokenizer_in.vocab_size + 2
    sos_token_input = [num_words_inputs - 2]
    eos_token_input = [num_words_inputs - 1]

    num_words_output = tokenizer_out.vocab_size + 2
    sos_token_output = [num_words_output - 2]
    eos_token_output = [num_words_output - 1]

    # Get the predicted sequence for the input sentence
    output = predict(model, sentence, tokenizer_in, tokenizer_out, sos_token_input, eos_token_input, sos_token_output, eos_token_output, MAX_LENGTH).numpy()
    
    # Transform the sequence of tokens to a sentence
    predicted_sentence = tokenizer_out.decode(
        [i for i in output if i < sos_token_output]
    )

    return predicted_sentence