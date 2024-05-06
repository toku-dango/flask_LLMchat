from flask import Flask, render_template, session
from flask_socketio import SocketIO, emit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.cache import InMemoryCache
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = "kazuma"  # このキーはセキュアなものに置き換えてください
app.config.from_object('config.config')
socketio = SocketIO(app)

# LangChain and memory setup
langchain_cache = InMemoryCache()
memory = ConversationBufferMemory(return_messages=True, k=5)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=app.config.get('GOOGLE_API_KEY'))

chain = ConversationChain(
    llm=llm,
    memory=memory
)

def init_chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

@app.route('/')
def index():
    init_chat()  # Initialize chat history
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    init_chat()
    user_input = data['text']
    user_message = HumanMessage(content=user_input, role="user")
    session['chat_history'].append(user_message)

    # Generate reply using LangChain
    response = chain(user_input)
    reply = response["response"]
    assistant_message = HumanMessage(content=reply, role="assistant")
    session['chat_history'].append(assistant_message)

    # Emit reply to the client
    emit('reply', {'text': reply})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
