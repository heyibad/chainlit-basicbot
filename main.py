import chainlit as cl
import os
from dotenv import load_dotenv, find_dotenv
from agents import AsyncOpenAI, RunConfig, Runner, OpenAIChatCompletionsModel,Agent

# Load the .env file
load_dotenv(find_dotenv()) 

# Get the API key from the .env file
gemini_api_key=os.getenv("GEMINI_API_KEY")


if not gemini_api_key:
    raise Exception("Gemini API Key not found in .env file")

# Create an instance of the OpenAI client for external LLM(Gemini)
client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Create an instance of the external client, for model
model = OpenAIChatCompletionsModel(
    openai_client=client,
      model="gemini-2.0-flash"
)

# Add a run configuration
config = RunConfig(
    model=model,
    model_provider= client,
    tracing_disabled=True
)

agent= Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model
)



@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello, I am a helpful assistant. How can I help you today?").send()


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role":"user", "content": message.content})
    result = await Runner.run(agent,run_config=config, input=history)
    history.append({"role":"assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()
   
