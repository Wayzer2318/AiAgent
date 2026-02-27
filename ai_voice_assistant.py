from urllib import request
from langchain_core import prompts
from requests import Response
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate, chat
import langchain_ollama
import subprocess
import os
import sys

# Suppress ALSA/JACK noise
devnull = open(os.devnull, "w")
os.dup2(devnull.fileno(), 2)  # redirect stderr to /dev/null


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
def listen():
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None


def speak(text):
    engine.say(text)
    engine.runAndWait()


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
