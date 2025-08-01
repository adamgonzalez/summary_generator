import numpy as np
import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from matplotlib import pyplot as plt


################################################################################
#####       Parsing the arXiv data for today
################################################################################
# Testing with a limited sample (2000 is the max on arXiv)
stop = 2000

# Define the subject list and short forms
subject_list =  ['Astrophysics of Galaxies (astro-ph.GA)', 
                 'Cosmology and Nongalactic Astrophysics (astro-ph.CO)', 
                 'Earth and Planetary Astrophysics (astro-ph.EP)',
                 'High Energy Astrophysical Phenomena (astro-ph.HE)',
                 'Instrumentation and Methods for Astrophysics (astro-ph.IM)',
                 'Solar and Stellar Astrophysics (astro-ph.SR)']
short_list =    ['GA', 'CO', 'EP', 'HE', 'IM', 'SR']

# Grab the new papers for today and parse their metadata
list_of_new_papers = requests.get('https://arxiv.org/list/astro-ph/new')
soup = BeautifulSoup(list_of_new_papers.content, 'html.parser')

# Grab paper hyperlinks
links = np.array([])
i = 0
for item in soup.find_all('a', attrs={'href': re.compile("^/abs/")}):
    links = np.append(links, f"https://arxiv.org{item.get('href')}")
    i += 1
    if (i == stop):
        break

# Get paper subjects
subjects = np.array([])
i = 0
for item in soup.find_all('span', attrs={'class': 'primary-subject'}):
    subjects = np.append(subjects, item.string)
    i += 1
    if (i == stop):
        break

# Grab paper metadata
paper_metadata = soup.find_all('div', class_='meta')


################################################################################
#####       Setting up the LLM 
################################################################################
# # Define the LLM and summary function with prompt --- DEPRECATED
# from langchain_ollama import OllamaLLM
# llm = OllamaLLM(model = "llama3.2:3b")
# def summarize_abstract(abstract):
#     prompt = f"Write a one sentence summary of the following text. Only include information that is part of the text. Do not indicate that you are giving a summary.\n\n{abstract}\n\nSummary:"
#     response = llm(prompt)
#     return response

# Define the prompt
prompt_template = """
Write a one sentence summary of the following text. 
Only include information that is part of the text. 
Do not indicate that you are giving a summary.
Do not provide your own opinion or analysis.
{abstract}
Summary:
"""
prompt = PromptTemplate.from_template(prompt_template)

# Define the LLM
llm = ChatOpenAI(
    temperature     = 0.0,
    model_name      = "llama3.2:3b",
    openai_api_key  = "ollama",
    openai_api_base = "http://localhost:11434/v1"
)
llm_chain = prompt | llm


################################################################################
#####       Defining several useful functions
################################################################################
# Define the summarizer function
def summarize_abstract(abstract):
    response = llm_chain.invoke(abstract)
    return response.content

# Define the 'wrapper' for the summarizer 
def get_summaries(category, metadata, numabs):
    summaries = np.array([])
    i = 0
    c = 0
    for entry in metadata:
        if (entry.find('span', attrs={'class': 'primary-subject'}).string == category):
            bar_placeholder.progress(c/numabs, text=f"...working on {c+1}/{numabs}")
            summaries = np.append(summaries, summarize_abstract(entry.find('p').text.strip()))
            c += 1
        i += 1
        if (i == stop):
            break
    bar_placeholder.progress(numabs/numabs)
    return( summaries )

# Define the distribution plotting function
def plot_dist(subject):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    if len(subject) == 1:
        ax.bar(short_list, all_nums, facecolor='#deebf7')
        ax.bar(short_list[subject[0]], all_nums[subject[0]], facecolor='#3182bd')
    else:
        ax.bar(short_list, all_nums, facecolor='#3182bd')
    ax.set_xlabel('astro-ph')
    ax.set_ylabel('Number of papers')
    ax.yaxis.get_major_locator().set_params(integer=True)
    return fig


################################################################################
#####       Setting up the streamlit app
################################################################################
# Creating lists based on subjects
GA_idx = np.where(np.array(subjects) == subject_list[0])[0]
CO_idx = np.where(np.array(subjects) == subject_list[1])[0]
EP_idx = np.where(np.array(subjects) == subject_list[2])[0]
HE_idx = np.where(np.array(subjects) == subject_list[3])[0]
IM_idx = np.where(np.array(subjects) == subject_list[4])[0]
SR_idx = np.where(np.array(subjects) == subject_list[5])[0]
all_idxs = np.array([GA_idx, CO_idx, EP_idx, HE_idx, IM_idx, SR_idx], dtype=object)
all_nums = np.array([len(GA_idx), len(CO_idx), len(EP_idx), len(HE_idx), len(IM_idx), len(SR_idx)], dtype=int)

# Initialize the streamlit app page
st.set_page_config(layout = 'wide')
st.title("Daily arXiv abstract summary generator", width = 'content')
distribution, abstracts = st.columns(2)

# Plot the full distribution of papers per subject
with distribution:
    st.write(f'There are {len(links)} new arXiv papers for today.')
    plot_placeholder = st.empty()   # create a placeholder for the plot
    plot_placeholder.pyplot(plot_dist(np.arange(6)))

with abstracts:
    # Create a drop-down menu of subjects for the user to select one from
    selection_list = np.append(' ', subject_list)
    result = st.selectbox('Select a category to get one-sentence abstract summaries for:', selection_list)
    if (result == selection_list[0]):
        # Give a blank option that doesn't run anything, but replots the full distribution
        st.write(' ')
    if (result != selection_list[0]):
        # If a subject (not blank) was selected: replot the distribution, get the links, and summarize only the selected abstracts
        idx = np.where(result == np.array(subject_list))[0]
        plot_placeholder.empty()
        plot_placeholder.pyplot(plot_dist(idx))
        relevant_links = links[all_idxs[idx][0]]
        with st.spinner(text = "Summarizing abstract(s)...", show_time = True):
            bar_placeholder = st.empty()    # create a placeholder for the progress bar
            summaries = get_summaries(result, paper_metadata, len(all_idxs[idx][0]))
            bar_placeholder.empty()
        st.success('Done!')
        for j in range(0, len(summaries)):
            # Output the abstract summaries and their links
            if (j > 0):
                st.divider()
            st.write(f"{summaries[j]}")
            st.link_button("arXiv link", f"{relevant_links[j]}")
