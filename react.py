# Abhishek Dutta, Copyright 2026, MIT License.

import streamlit as st; import os; os.environ['KERAS_BACKEND'] = 'jax'
import keras; from keras import ops
import keras_hub as kh; import kagglehub
import wikipedia, json, re; from ddgs import DDGS
import tensorflow as tf; import time
if tf.config.list_physical_devices('GPU'):
  keras.mixed_precision.set_global_policy('mixed_float16')

st.title('Nano Foundation Model')
os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle_username']
os.environ['KAGGLE_KEY'] = st.secrets['kaggle_key']

# load the model once and use it across all users and sessions
@st.cache_resource
def load_model(): return kagglehub.model_download('abhionic/agent/keras/50m')

path = load_model()
model = keras.saving.load_model(f'{path}/model.keras', compile=False)
sampler = kh.samplers.GreedySampler() #TopPSampler(temperature=1.0, p=0.1, k=5)
model.compile(sampler=sampler); tok = model.preprocessor.tokenizer
# architecture patching
tok.start_token_id = None; tok.end_token_id = tok.token_to_id('<eos>')
tok.pad_token_id = tok.token_to_id('<pad>'); tok.start_of_image_token_id = None

# initialize chat history
if 'messages' not in st.session_state: st.session_state.messages = []

# stream assistant response in chat message container
def stream(outext):
  def stream_data():
    for word in outext.split(' '): yield word + ' '; time.sleep(0.02)
  with st.chat_message('assistant'): st.write_stream(stream_data)

# display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']): st.markdown(message['content'])

#ReAct orchestration
def get_relevant_snippet(text, query, max_chars=250):
    # Split into sentences safely
    sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
    query_terms = set(re.findall(r'\w+', query.lower()))

    # Score sentences based on query term overlap
    scored = []
    for s in sentences:
        s_terms = set(re.findall(r'\w+', s.lower()))
        score = len(query_terms & s_terms); scored.append((score, s))

    # Sort by score (highest first) and take the best 2 sentences
    scored.sort(key=lambda x: x[0], reverse=True)
    best_text = " ".join([s for score, s in scored[:2]])

    return best_text[:max_chars] if best_text else text[:max_chars]

def search_wiki(query): # wikipedia tool
    try:
        # fetch more text initially (e.g., 5 sentences instead of 2)
        raw_text = wikipedia.summary(query, sentences=5, auto_suggest=True)
        return get_relevant_snippet(raw_text, query)
        #return wikipedia.summary(query, sentences=2, auto_suggest=True)[:200]
    except Exception as e: return f"Search failed: {e}"

def search_duck(query): # duckduckgo tool
    try:
        with DDGS() as ddgs:
            # fetch top 3 results instead of 1 to increase chances of finding the answer
            results = list(ddgs.text(query, max_results=3))
            if results:
                combined_text = " ".join([r.get('body', '') for r in results])
                return get_relevant_snippet(combined_text, query)
                #return results[0].get('body', results[0].get('title', ''))[:200]
        return "No exact answer found."
    except Exception as e: return f"Search failed: {e}"

def calc(expr): # calculator tool
    try:
        clean_expr = expr.replace(" ", "") # remove tokenizer artifact spaces
        return str(eval(clean_expr))
    except Exception as e: return f"Error: {e}"

def react_run(question, max_steps=3):
    text = f'<|User|> {question} <|End|>'; full = ""

    # Precompute special token IDs for stopping
    act_end_id = tok.token_to_id('<|/Act|>'); end_id = tok.token_to_id('<|End|>')

    for step in range(max_steps):
        # Generate text
        out = model.generate(text, max_length=model.preprocessor.sequence_length, 
                             stop_token_ids=[act_end_id, end_id])
        gen_text = out[len(text):]; text = out

        # Print thought process if available
        if '<|Think|>' in gen_text and '<|/Think|>' in gen_text:
            thought = gen_text[gen_text.find('<|Think|>')+9 : gen_text.find('<|/Think|>')].strip()
            response = f"Step {step + 1} Thought: {thought}"; stream(response); full += response

        # Case 1: Model generated an <|Act|> block
        if '<|/Act|>' in gen_text:
            if '<|Act|>' in gen_text:
                act_content = gen_text[gen_text.find('<|Act|>')+7 : gen_text.find('<|/Act|>')].strip()
                response = f"Step {step + 1} Action: {act_content}"; stream(response); full += response

                # Execute the parsed tool/function
                if act_content.startswith('search'):
                    query = act_content[act_content.find('[')+1:act_content.find(']')].strip()
                    obs = search_duck(query)
                elif act_content.startswith('calc'):
                    expr = act_content[act_content.find('[')+1:act_content.find(']')].strip()
                    obs = calc(expr)
                else: obs = "Invalid action format."

                # Force the model to answer on the final step by altering the observation
                if step == max_steps-2: obs += " Provide the final answer now."
                # Append observation to context for the next loop
                response = f"Observation: {obs}\n"; stream(response); full += response 
                text += f" <|Observe|> {obs} <|/Observe|>"

        # Case 2: Model generated an <|Answer|> and <|End|>
        elif '<|End|>' in gen_text:
            if '<|Answer|>' in gen_text and '<|/Answer|>' in gen_text:
                ans = gen_text[gen_text.find('<|Answer|>')+10 : gen_text.find('<|/Answer|>')].strip()
                response = f"Answer: {ans}"; stream(response); full += response
            return full

    response = "Reached max steps."; stream(response); full += response
    return full

# react to user input
if prompt := st.chat_input('please enter your query'):
    # add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    # display user message in chat message container
    with st.chat_message('user'): st.markdown(prompt)
    full_response = react_run(prompt)
    # add assistant response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': full_response})
