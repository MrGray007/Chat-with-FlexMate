import os
from dotenv import load_dotenv
load_dotenv()
#os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['LANGSMITH_API_KEY']=os.getenv('LANGSMITH_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2']='true'
os.environ['LANGSMITH_PROJECT']='Gym_Buddy'
#os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import streamlit as st
st.title('Welcome to FlexMate 💪,Your Gym Companion')
# groq_api_key=st.text_input('Enter the Groq API Key :',type='password',key='groq_key')
# HF_api_key=st.text_input('Enter the HuggingFace API Key :',type='password',key='HF_key')
st.sidebar.title("🔑 API Keys")

groq_api_key = st.sidebar.text_input(
    "Enter the Groq API Key:", type="password", key="groq_key"
)
hf_api_key = st.sidebar.text_input(
    "Enter the HuggingFace API Key:", type="password", key="hf_key"
)
if groq_api_key and hf_api_key:
    os.environ['HF_TOKEN']=hf_api_key
    embedder=embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}   # force CPU
    )
    print('emb done')
    lib=FAISS.load_local(r'D:\GenAI\GenAi_Pr\online_gymbuddy\new_gym_vec',embedder,allow_dangerous_deserialization=True)
    lib_ret=lib.as_retriever()
    print('vector loaded done')

    llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0.0,max_retries=2,max_tokens=1000,api_key=groq_api_key)
    print('LLM Loaded done')

    #Message history
    contextualize_q_system_prompt=(
        'Given a chat history and the latest user question'
        'which might reference context in the chat history,'
        'formulate a standlone question which can be understood'
        'without the chat history.Do not answer the question'
        'just formulate it if needed and otherwise return it as is.'
    )
    q_prompt=ChatPromptTemplate.from_messages([
        ('system',contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{input}'),])

    history_aware=create_history_aware_retriever(llm|StrOutputParser(),lib_ret,q_prompt)

    system_prom='You are a motivational GYM COACH 🏋️‍♂️'
    # user_prom='''Take the following factual answer and rephrase it in a supportive,
    # energetic, gym-buddy tone. Add encouragement where possible.
    # <context>
    # {context}
    # <\context>'''
    user_prom="""Use the following context to answer the user.
        Be motivational and supportive like a gym buddy!

        <context>
        {context}
        </context>

        Question: {input}
        """
    info_prom=ChatPromptTemplate.from_messages([('system',system_prom),MessagesPlaceholder('chat_history'),('user',user_prom)])
    info_document_chain=create_stuff_documents_chain(llm,info_prom)


    retrieval_chain=create_retrieval_chain(lib_ret,info_document_chain)
    def get_session_history(seesion_id):
        if seesion_id not in st.session_state['store']:
            st.session_state['store'][seesion_id]=ChatMessageHistory()
        return st.session_state['store'][seesion_id]

    con_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key='input',       # user input
        history_messages_key='chat_history',
        output_messages_key='answer'  # variable name in your prompt
    )

    # supportive,
    # energetic

    motivational_prom="""
    You are a motivational GYM COACH 🏋️‍♂️.
    Take the following factual answer and rephrase it in a {mood}, gym-buddy tone. Add encouragement where possible.

    Answer:
    <context>
    {context}
    </context>
    Now rephrase:
    """

    rephrase_prom=ChatPromptTemplate.from_messages([('user',motivational_prom)])
    motivation_llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0.6,max_retries=2,api_key=groq_api_key,max_tokens=1000)

    moti_response=create_stuff_documents_chain(motivation_llm,rephrase_prom)
    print('Motivation LLM done')

    def gym_coach_pipeline(input_text:str,session_id:str)->str:
        moood={'supportive':'supportive,energetic,super cool','angry':'angry,strictly,hard coach'}
        res=con_rag_chain.invoke({'input':input_text},config={'configurable':{'session_id':session_id}})
        res=Document(page_content=res['answer'])
        rephrased_res=moti_response.invoke({"context":[res],'mood':'supportive,energetic,super cool,enthusiastic,and energetic'})
        return rephrased_res
    
    session_id=st.text_input('Enter your session id',value='user_1')
    if 'store' not in st.session_state:
        st.session_state['store']={}
    # input_text=st.text_input('What question you have in Mind????')
    #mood=st.multiselect('Coach Mood',['supportive','angry'],default=['supportive'])
    if session_id in st.session_state['store']:
        st.write("### 💬 Chat History")
        for msg in st.session_state['store'][session_id].messages:
            if msg.type == "human":
                st.markdown(f"**You:** {msg.content}")
            else:
                st.markdown(f"**Coach:** {msg.content}")
    def submit_input():
        st.session_state['user_input'] =''
    input_text=st.text_area("💬 Ask your question:", key="user_key",on_change=submit_input)
    if input_text:
        st.write(gym_coach_pipeline(input_text,session_id))