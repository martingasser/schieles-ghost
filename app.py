import streamlit as st
from transformers import pipeline
import tokenizers

model_selectbox = st.sidebar.selectbox(
    'Choose a model',
    ('Schiele',)
)

length_slider = st.sidebar.slider(label='Text length', min_value=100, max_value=1000)

num_beams = st.sidebar.slider(label='#Beams', min_value=1, max_value=10, step=1)

no_repeat_ngram_size = st.sidebar.slider(label='No repeat ngram size', min_value=0, max_value=3, step=1)

do_sample = st.sidebar.checkbox(label='Do sample', value=True)

top_k = st.sidebar.slider(label='Top k', min_value=0, max_value=100, step=1)

temperature = st.sidebar.slider(label='Temperature', min_value=0.01, max_value=2.0, value=1.0)

names = open('schiele_names.txt')
names = names.readlines()

#, 'German recipes'

st.title(model_selectbox)

name = st.selectbox('Choose a character to start with', names)

@st.cache(allow_output_mutation=True)
def get_pipeline():
    if model_selectbox == 'German recipes':
        return pipeline('text-generation',model='./gpt2-gerchef', tokenizer='anonymous-german-nlp/german-gpt2', device=0)
    elif model_selectbox == 'Schiele':
        return pipeline('text-generation',model='./gpt2-schiele', tokenizer='anonymous-german-nlp/german-gpt2', device=0)

with st.form(key='my_form'):
    name = name.rstrip()
    text_input = st.text_input(label=name + ' says:', value='')
    submit_button = st.form_submit_button(label='Generate')

    if submit_button:
        generator = get_pipeline()
        
        result = generator(
            name + ':' + text_input,
            max_length=length_slider,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature)[0]['generated_text']
        lines = result.split('\n')

        for i, line in enumerate(lines):
            if ':' in line:
                p = line.find(':')
                speaker = line[0:p]
                txt = line[p+1:]
                txt = txt.replace('`', '')
                st.markdown(f'__{speaker}__: {txt}')
            #else:
            #    st.markdown(line)
