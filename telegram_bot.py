import os
import streamlit as st
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA, LLMChain, LLMMathChain
from langchain.agents import Tool, initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from utils.loadVectorStore import load_vectorstore
import os
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv(dotenv_path='app.env')


# print("Starting Telegram Bot...", os.getenv("GROQ_API_KEY"))

# llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=st.secrets["groq_api_key"])



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
    description="Use this tool ONLY for direct mathematical calculations or expressions (e.g., '2+2', 'sqrt(16)'). Do NOT use for algebraic equations or word problems."
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
You are a agent tasked for solving users mathematical questions. Use the following context to answer the question in detail.
Context: {context}.
Provide a detailed explanation with step by step solution for the question below as given in the example:
Question:{question}
Answer the question in a step-by-step manner, following these steps:
1. Analyze the problem type.
2. Choose the appropriate solution method.
3. Show step-by-step solution.
4. Clearly state the final answer.

Example: 90% of 136 ?
Answer:
To find 90% of 136, we can use the formula:
Percentage = (Part / Whole) * 100
Part = (Percentage * Whole) / 100
Part = (90 * 136) / 100
Part = 122.4
Hence 90% of 136 is 122.4.

"""

prompt_template=PromptTemplate(
    input_variables=["context", "question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="Use this tool only for elementary-level math word problems that require reasoning, but not direct calculation or retrieval from PDFs."
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm your MathBot. Ask me anything from your PDFs?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conversation_history = []
    query = update.message.text
    conversation_history.append({"role": "user", "content": query})
    await update.message.reply_text("Thinking...")
    try:
        answer = assistant_agent.run(conversation_history)
        answer = answer.strip() + '\n\nFor more detailed solution, please visit below link: \n\n https://mathbot-problem-solver.streamlit.app/'
        conversation_history.append({"role": "assistant", "content": answer})
        # If the answer is a list, convert it to a string
        if isinstance(answer, list):
            answer = "\n".join(answer)
        elif isinstance(answer, dict):
            answer = "\n".join(f"{k}: {v}" for k, v in answer.items())
    except Exception as e:
        answer = f"Error: {str(e)}"
    await update.message.reply_text("Ans: "+ answer)

def main():
    app = ApplicationBuilder().token(st.secrets["TELEGRAM_BOT_TOKEN"]).build()
    # app = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()



if __name__ == "__main__":
    main()
