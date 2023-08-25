"""
Authoring notes
"""

import openai 
import pandas as pd 
import polars as ps 
import langchain
from langchain.llms import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os 

# Retrieve OPENAI_API_KEY
#load_dotenv()
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize our LLM 
llm = OpenAI(model_name="gpt-4")
chat_model = ChatOpenAI(model_name="gpt-4")

# Load documents
loader = PyMuPDFLoader("./papers/sparsemodels.pdf")
documents = loader.load()

# Prompt Template Creation (Working)
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are acting as Ronald McDonald the clown who is the face of McDonalds. Please only respond with things that Ronald would say in his daily life.
    If you don't know how Ronald McDonald would act, then simply respond to my queries with things that would be socially acceptable for a clown to say. That
    would be good enough. Thank you! 

    Question: {question}
    """
)
#print(chat_model.predict(prompt.format(question="How do you respond to a group of kids who want your autograph?")))

# Initializing Memory for Conversation
memory_prompt = PromptTemplate(
    input_variables=["prior_pdf", "pdf"],
    template="""
    {prior_pdf}

    PDF: {pdf}
    """
)

memory = ConversationBufferMemory(memory_key="prior_pdf")

# Format LLMChain

llm_chain_pdf = LLMChain(
    llm=chat_model,
    prompt=memory_prompt,
    verbose=True,
    memory=memory,
)

### This is the format for entering each of the documents within the PDF loader. 
#llm_chain.predict(pdf=document[x])

# Initialize Pre-Prompt

pre_prompt="""
    Hello, you are going to be my researcher water-down buddy! Meaning, you'll be given an entire research paper 
    broken down into multiple pages, and after you digest the pages, I would like you to give me a full-length summary of the entire
    paper in an ELI5 (Explain Like I'm 5) manner. 

    PLEASE ONLY RESPOND BACK WITH "OKAY" AFTER THIS MESSAGE AND EACH OF THE PAGES OF THE PDF I'LL PROVIDE TO YOU. 

    After each of the pages, I'll ask you a prompt, which asks to summarize everything you've read, and then you'll provide the explanations. 
    I would like the summaries to be very clearly digestible to anyone, even a 5 year old (hence ELI5), and you'll continue generating output until 
    you come to your final conclusions. 

    REMEMBER, IF YOU DO NOT FINISH YOUR SUMMARY AND THE TOKEN LIMIT ARISE FOR YOU, THEN SIMPLY ALLOCATE ENOUGH TOKENS TO ADD "CONTINUING..." AT THE 
    END OF THE PARTICULAR RESPONSE. 

    This will ensure that I provide a one-word input to you to continue your summary until completion. 

    IF YOU UNDERSTAND, SAY "OKAY", BUT IF YOU HAVE ANY FURTHER QUESTIONS, SPEAK NOW! Thank you!
    """

print(chat_model.predict(pre_prompt), "\n\n")

# Loop through documents
inter_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
        THIS IS THE MIDDLE OF THE PDF TEXT INPUT! INPUT IS PROVIDED BELOW. PLEASE RESPOND WITH ONLY "OKAY" ONCE YOU'VE DIGESTED
        THIS PARTICULAR PAGE OF THE PDF. ONLY RESPOND WITH "OKAY" IF YOU ALSO REMEMBER THE PREVIOUS INPUT AS WELL. IF YOU DO NOT, 
        THEN TRY RESPONDING WITH SOMETHING OTHER THAN "OKAY". 

        {text}
    """
)
for document in documents: 
    print(chat_model.predict(inter_prompt.format(text=document)), "\n\n")

end_prompt = """
    THIS IS THE END OF THE PDFs. PLEASE RESPOND WITH THE SUMMARY OF THE ENTIRE PDF IN AN ELI5 SENSE! 
    IF YOU WISH TO CONTINUE EXPLAINING BUT YOU DON'T HAVE ANY MORE TOKENS LEFT FOR OUTPUT, THEN RESERVE ENOUGH TOKENS
    TO ADD "CONTINUING..." AT THE END OF THE OUTPUT, SO THAT I CAN LOOP THIS CHAT AND LET YOU CONTINUE EXPLAINING. 

    ONCE YOU'RE FULLY DONE WITH EXPLAINING, RESERVE SOME TOKENS FOR OUTPUT AT THE END FOR THE WORDS "FINISHED" SO THAT 
    I CAN STOP THE CONVERSATION AND RECORD ALL OF YOUR SUMMARIZED OUTPUTS.
"""

print(chat_model.predict(end_prompt), "\n\n")




# Output of the LLM based on prompt
#print(llm(prompt.format(question="How many eggs would you say would fill you right up? Calories, protein, and all things considered within your explanation.")))








