import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from utils.loadVectorStore import load_vectorstore
from dotenv import load_dotenv
load_dotenv(dotenv_path='app.env')



## Set upi the Stramlit app
st.set_page_config(page_title="Math Problem Solver",page_icon="🧮")
st.title("Your Personal Maths Problems Solver")

# groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")


# if not groq_api_key:
#     st.info("Please add your Groq API key to continue")
#     st.stop()

llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=st.secrets['groq_api_key'])
# llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=os.getenv("GROQ_API_KEY"))

## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Web to find various information on the topics mentioned"

)

## Initialize the Math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="Use this tool ONLY for direct mathematical calculations or expressions (e.g., '2+2', 'sqrt(16)') and share solutions as well." \
    " Do NOT use for algebraic equations or word problems."
)


#Loading the VectorStore here:
vectorstore_path = './Maths_Datasets/pdf_vectorstore'
vectorstore = load_vectorstore(vectorstore_path, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

#Custom Trained Tool on NCERT Maths Books:
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

custom_data_tool = Tool(
    name="Custom Data MathBot",
    func=lambda q: qa_chain.run(q),
    description="Use this tool for questions that are related to the content of the uploaded NCERT Maths PDFs, including step-by-step solutions and explanations."
)

prompt="""
You are a agent tasked for solving users mathematical questions. Provide a 
detailed explanation with step by step solution for the question below using any of the tools provided:
Question:{question}
Answer:

"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="Use this tool for elementary-level math word problems that require reasoning, but not direct calculation or retrieval from PDFs."
)

#Initialize the agents
assistant_agent=initialize_agent(
    # tools=[wikipedia_tool,calculator,reasoning_tool,custom_data_tool],
    tools=[custom_data_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a Math Chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## Lets start the interaction
question=st.text_area("Enter your question:","").replace("?","")
st.markdown(
    """
    <style>
    textarea {
        font-size: 2rem !important;
    }
    input {
        font-size: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("Get Answer"):
    if question:
        with st.spinner("Generating Response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('# Response:')
            st.success(response)

    else:
        st.warning("Please enter the question?")