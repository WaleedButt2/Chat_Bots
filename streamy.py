import streamlit as st
from ChatBot import SimplifiedChatbot
from streamlit_chat import message
import logging
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='Logs/Queries.log'  # Specify the log file name and path
)
if "new_user" not in st.session_state:
    st.session_state["new_user"]=True
    st.session_state['input_text'] = True
    # print("Call")
    # logger.add("file_{time}.log", rotation="50 MB")

if "model_initialized" not in st.session_state:
    # st.write("Calling again")
    st.session_state["model_initialized"] = True
    bot = SimplifiedChatbot()
    st.session_state["bot"]=bot
else:
    # st.write("In else modal already loaded")
    bot = st.session_state["bot"]
st.title("Tax Info Chatbot- V-1.5")
st.write("Compatible for Vat and CT using Heirarchy")


# bot = ChatbotResponse()

# else:
#     st.write(user_data[0])
#     bot.set_user("1")
#     st.session_state["current_user"]=user_data[0]

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Welcome Simpla.ia Tax chatbot. How may I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:

        with st.spinner("Please wait ...."):
            logging.info(f"\nACTUAL QUERY : \n{query}")
            # logging.info(refined_query)
            response = bot.get_response(query)
            logging.info(f"\nBOT RESPONSE : \n{response}")
            st.write("# Sources Include")
            # st.write(bot.memory.load_memory_variables({}))
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
        # query=""
        # st.write(bot.prompt)


with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
