import os
import streamlit as st
import pandas as pd
import numpy as np
import json

# Load the JSON file
with open('understanding_to_decision_internvl.json', 'r') as file:
    data = json.load(file)
    
# the data comes as a list of dictionaries, with entries by: {'keypoint_text_desc': keypoint_text_desc, 'keypoint_gt': keypoint_gt, 'gpt4o_raw_pred': keypoint,'gpt4o_pred':marked_part, 'image_id': image_id, 'category': category}
# Parse into a pandas dataframe

df = pd.DataFrame(data)
# Streamlit app

st.title('Mark Reasoning Results')

# summary statistics - mean of the decision column
st.write('**Summary statistics**')
st.write('GPT-4o Accuracy', df['gpt4o_decision'].mean())
st.write('Internvl Accuracy', df['internvl_decision'].mean())

# show accuracy of grouped by category
st.write('**Accuracy by keypoint**')
gpt4o_res = df.groupby('keypoint_text_desc')['gpt4o_decision'].mean()
internvl_res = df.groupby('keypoint_text_desc')['internvl_decision'].mean()
# show in single table
res = pd.concat([gpt4o_res, internvl_res], axis=1)
res.columns = ['GPT-4o', 'Internvl']
st.write(res)

gpt4o_results_path = 'gpt4o_mark_understanding_results'

# index selector slider
index = st.slider('Select an index:', 0, len(df) - 1, key='slider')

# show the image + relevant data: keypoint_text_desc, gpt4o_raw_pred, gpt4o_pred
keypoiint_text_desc = df.loc[index, 'keypoint_text_desc']
image_id = df.loc[index, 'image_id']
category = df.loc[index, 'category']
decision = df.loc[index, 'gpt4o_decision']
internvl_decision = df.loc[index, 'internvl_decision']
image_path = f"{gpt4o_results_path}/{image_id}/{keypoiint_text_desc}/image_with_mark.jpg"
st.image(image_path, caption=f'{category}, {keypoiint_text_desc}', use_column_width=True)
st.write('**LLM-based GPT-4 decision**:', decision)
st.write('**LLM-based Internvl decision**:', internvl_decision)
# use qoute or block to print the reasoning
with st.container(border=True):
    st.write('**GPT-4 Output (raw)**:', df.loc[index, 'gpt4o_raw_pred'])
    st.write('**InternVL Output (raw)**:', df.loc[index, 'internvl_raw_pred'])
# st.write('**GPT-4 Output (raw)**:', df.loc[index, 'gpt4o_raw_pred'])
# st.write('**Extracted reasoning**:', df.loc[index, 'gpt4o_pred'])




