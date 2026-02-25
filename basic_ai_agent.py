from os import truncate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core import prompts
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# initialize memory
chat_history = ChatMessageHistory()

# load model
llm = OllamaLLM(model="mistral")

# define ai prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="previous conv: {chat_history}\nUser: {question}\nAi:",
)


# ai fonction to run chat memory
def runchain(question):
    # retrive chat history manually
    chat_history_text = "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages]
    )

    response = llm.invoke(
        prompt.format(chat_history=chat_history_text, question=question)
    )

    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)

    return response


print("chatbot with memory")
print("exit to stop")
while True:
    user_input = input("you:")
    if user_input.lower() == "exit":
        break
    ai_response = runchain(user_input)
    print(ai_response)
# print("\n welcome to yout ai agents !")
#
# while True:
#     question = input(" what is your question ( exit to stop) : ")
#     if question.lower() == "exit":
#         print("goodbye")
#         break
#     response = llm.invoke(question)
#     print(response)
