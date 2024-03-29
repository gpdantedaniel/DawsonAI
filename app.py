import streamlit as st
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def generate_response(txt):
    # Instantiate the model
    model = AutoModelForSeq2SeqLM.from_pretrained("dantedgp/question-generator")
    tokenizer = AutoTokenizer.from_pretrained("dantedgp/question-generator")

    input_ids = tokenizer(txt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=100, do_sample=False)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return question

st.set_page_config(page_title='AI Quiz Generator')
st.title('AI Quiz Generator')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)