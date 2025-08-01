# summary_generator
A web app that grabs the abstracts of new daily submissions to astro-ph on arXiv.org and summarizes them in one sentence using a local LLM to enable a quick yet detailed evaluation of each publication. With >100 new submissions most days, this tool aims to help speed-up a routine task for astronomers. 

## Requirements
This is a Python-based web app the depends on the following packages:
* `bs4`
* `langchain_core`
* `lagnchain_openai`
* `numpy`
* `matplotlib`
* `requests`
* `streamlit`

The summaries are produced by an LLM. In the current version of the code, this is implemented by downloading one locally using `ollama`, namely the `llama3.2:3b` model by Meta as it is small enough (2GB) to test things with while also not requiring high-end hardware to run efficiently. 

## Usage
Simply install the required packages (with `pip` or `conda` depending on your setup) and desired LLM (edit the model name accordingly in the code if you do not use `llama3.2:3b`) and either download the `app_streamlit.py` file or clone the repository. 

The web app can be opened using `streamlit run app_streamlit.py`, which will bring you to a page consisting of a distribution of new submissions by astro-ph subject and a drop-down menu that will allow you to select a subject to produce abstract summaries for. Once the summaries have been generated, they will be posted alongside direct links to the corresponding articles hosted on arxiv.org. See the GIF below for a demonstration.
