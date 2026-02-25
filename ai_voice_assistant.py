from urllib import request
from langchain_core import prompts
from requests import Response
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate, chat
import langchain_ollama


# loading model
llm = langchain_ollama.OllamaLLM(model="mistral")

# initialize chat history
chat_history = ChatMessageHistory()

# initialize txt to speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 200)  # for the speed

# initialize recognizer
recognizer = sr.Recognizer()


# speak func
def speak(text):
    engine.say(text)
    engine.runAndWait()


# listen func
def listen():
    with sr.Microphone() as source:
        print("listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_amazon(audio)
        print(f"you said : {query}")
    except sr.UnknownValueError:
        print(" i did not understand")
        return ""
    except sr.RequestError:
        print("speech rec not available")


# chat prompts

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="previous conv : {chat_history}\n user: {question}\n Ai:",
)

# process ai response


def run_chain(question):
    #  chat history
    chat_history_text = "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages]
    )
    # run ai generation
    response = llm.invoke(
        prompt.format(chat_history=chat_history_text, question=question)
    )
    # store ai and user input
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)

    return response


# main
print("hello")
while True:
    query = listen()
    if query and ("exit" in query or "stop" in query):
        speak("goodbye")
        break
    if query:
        response = run_chain(query)
        print(f"ai :{response}")
        speak(response)
