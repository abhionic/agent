# Abhishek Dutta, Copyright 2026, MIT License.

import streamlit as st; import os; os.environ['KERAS_BACKEND'] = 'jax'
import keras; from keras import ops
import keras_hub as kh; import kagglehub
import wikipedia, json, re; from ddgs import DDGS
import tensorflow as tf; import time
if tf.config.list_physical_devices('GPU'):
  keras.mixed_precision.set_global_policy('mixed_float16')

st.title('ReAct Agent')
os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle_username']
os.environ['KAGGLE_KEY'] = st.secrets['kaggle_key']

# initialize chat history
if 'messages' not in st.session_state: st.session_state.messages = []

# stream assistant response in chat message container
def stream(outext): 
    for word in outext.split(' '): yield word + ' '; time.sleep(0.02)
    with st.chat_message('assistant'): return st.write_stream(stream_data)

# display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']): st.markdown(message['content'])

# load the model once and use it across all users and sessions
@st.cache_resource
def load_model(): return kagglehub.model_download('abhionic/agent/keras/15m')

path = load_model()
model = keras.saving.load_model(f'{path}/model.keras')
vocab = f'{path}/vocab.txt'; seq_len = 512

# tokenizer
control_tokens = ['[PAD]', '[UNK]']
structure_tokens = ['<|User|>', '<|Model|>', '<|Think|>', '<|Act|>', '<|Observe|>',
  '<|Answer|>', '<|/Think|>', '<|/Act|>', '<|/Observe|>', '<|/Answer|>', '<|End|>']
reserved_tokens = control_tokens + structure_tokens
tokenizer = kh.tokenizers.WordPieceTokenizer(vocab, lowercase=True, strip_accents=True,
                        special_tokens=reserved_tokens, special_tokens_in_strings=True)
long_packer = kh.layers.StartEndPacker(seq_len, return_padding_mask=True)
sampler = kh.samplers.TopPSampler(temperature=1, p=0.1, k=5)
def next(prompt, cache, index): # compute logits
    logits = model(prompt)[:, index-1, :]
    hidden_states = None; return logits, hidden_states, cache

#ReAct orchestration
def get_relevant_snippet(text, query, max_chars=250):
    # Split into sentences safely
    sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
    query_terms = set(re.findall(r'\w+', query.lower()))

    # Score sentences based on query term overlap
    scored = []
    for s in sentences:
        s_terms = set(re.findall(r'\w+', s.lower()))
        score = len(query_terms & s_terms)
        scored.append((score, s))

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
        clean_expr = expr.replace(" ", "") # remove WordPiece artifact spaces
        return str(eval(clean_expr))
    except Exception as e: return f"Error: {e}"

def react_run(question, max_steps=3):
    text = f'<|User|> {question} <|End|>'; full = ""

    # Precompute special token IDs for matching
    act_start_id, act_end_id = tokenizer('<|Act|>')[0], tokenizer('<|/Act|>')[0]
    think_start_id, think_end_id = tokenizer('<|Think|>')[0], tokenizer('<|/Think|>')[0]
    ans_start_id, ans_end_id = tokenizer('<|Answer|>')[0], tokenizer('<|/Answer|>')[0]
    end_id = tokenizer('<|End|>')[0]

    # Helper to extract text between specific start/end tags from generated tokens
    def extract(tokens, start_id, end_id):
        s = ops.where(ops.equal(tokens, start_id))[0]
        e = ops.where(ops.equal(tokens, end_id))[0]
        if ops.size(s) > 0 and ops.size(e) > 0:
            return tokenizer.detokenize(tokens[int(s[0])+1 : int(e[0])]).strip()
        return ""

    for step in range(max_steps):
        # Prepare prompt tokens
        tokens, _ = long_packer(tokenizer(text))
        tokens = ops.expand_dims(tokens, axis=0)
        ct = ops.count_nonzero(tokens)

        # Generate next segment of tokens until an early stop (or max tokens)
        out = sampler(next=next, prompt=tokens, index=ct)
        padidx = ops.where(ops.equal(out, 0))
        out = out[0, :padidx[1][0]] if ops.size(padidx) > 0 else out[0]

        gen_tokens = out[ct:]
        act_end_idx = ops.where(ops.equal(gen_tokens, act_end_id))[0]
        end_idx = ops.where(ops.equal(gen_tokens, end_id))[0]

        # Case 1: Model generated an <|Act|> block
        if ops.size(act_end_idx) > 0:
            out = out[:ct + int(act_end_idx[0]) + 1]
            text = tokenizer.detokenize(out)

            thought = extract(gen_tokens, think_start_id, think_end_id)
            if thought: response = stream(f"Step {step+1} Thought: {thought}"); full += response

            act_content = extract(gen_tokens, act_start_id, act_end_id)
            response = stream(f"Step {step+1} Action: {act_content}"); full += response

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

            response = st.write(f"Observation: {obs}\n"); full += response
            # Append observation to context for the next loop
            text += f" <|Observe|> {obs} <|/Observe|>"

        # Case 2: Model generated an <|Answer|> and <|End|>
        elif ops.size(end_idx) > 0:
            out = out[:ct + int(end_idx[0]) + 1]
            # Update context for printing detokenization, optional for return

            thought = extract(gen_tokens, think_start_id, think_end_id)
            if thought: response = stream(f"Final Thought: {thought}"); full += response

            ans = extract(gen_tokens, ans_start_id, ans_end_id)
            if ans: response = stream(f"Answer: {ans}"); full += response
            return full

        # Edge case: generation stopped before an Act or End block
        else: text = tokenizer.detokenize(out)

    response = stream("Reached max steps."); full += response
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
