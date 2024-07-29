import os
from streamlit_chat import message
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI

def submit():
    st.session_state['input'] = st.session_state['query']
    st.session_state['query'] = ''

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Please set it in your .env file.")
        return

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    # Initialize session state for requests and responses
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    if 'responses' not in st.session_state:
        st.session_state['responses'] = []
    if 'query' not in st.session_state:
        st.session_state['query'] = ''
    if 'input' not in st.session_state:
        st.session_state['input'] = ''

    # File uploader for CSV files
    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        # Create a CSV agent
        try:
            agent = create_csv_agent(
                OpenAI(temperature=0, api_key=api_key), csv_file, verbose=True, allow_dangerous_code=True
            )
        except Exception as e:
            st.error(f"Failed to create CSV agent: {e}")
            return

        # Container for chat history
        response_container = st.container()
        # Container for text box
        textcontainer = st.container()

        with textcontainer:
            query = st.text_input("Ask a question about your CSV:", key='query', on_change=submit)
            if st.session_state['input']:
                with st.spinner("Processing..."):
                    response = agent.run(st.session_state['input'])
                st.session_state.requests.append(st.session_state['input'])
                st.session_state.responses.append(response)
                st.session_state['input'] = ''

        with response_container:
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    if i < len(st.session_state['requests']):
                        message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state['responses'][i], key=str(i))

if __name__ == "__main__":
    main()
