from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext, MessageFactory
from botbuilder.schema import Activity, ActivityTypes, Attachment, ConversationReference
from dotenv import load_dotenv
import os
import json
from agno.models.groq import Groq
from agno.agent import Agent
from agno.team import Team
from agno.vectordb.lancedb import LanceDb
from agno.embedder.fastembed import FastEmbedEmbedder
from agno.agent import AgentKnowledge
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from fastembed import TextEmbedding

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
APP_ID = os.getenv('APP_ID')
APP_PASSWORD = os.getenv('APP_PASSWORD')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = BotFrameworkAdapterSettings(app_id=APP_ID, app_password=APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

ADAPTIVE_CARD = {
    "type": "AdaptiveCard",
    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
    "version": "1.5",
    "body": [
        {
            "type": "TextBlock",
            "text": "Test Notification",
            "weight": "bolder",
            "size": "medium"
        },
        {
            "type": "TextBlock",
            "text": "Hey, you received a new message!",
            "wrap": True
        }
    ],
    "actions": [
        {
            "type": "Action.OpenUrl",
            "title": "Learn More",
            "url": "https://adaptivecards.io"
        }
    ]
}

CONVERSATION_REFERENCES = {}
REFS_FILE = "conversation_references.json"

def load_conversation_references():
    """Load conversation references from file."""
    global CONVERSATION_REFERENCES
    try:
        if os.path.exists(REFS_FILE):
            with open(REFS_FILE, 'r') as f:
                data = json.load(f)
                valid_refs = {}
                for key, val in data.items():
                    if key.startswith("msteams:") and val.get('conversation') and val['conversation'].get('id'):
                        valid_refs[key] = ConversationReference(**val)
                    else:
                        print(f"Skipping invalid or non-Teams reference {key}")
                CONVERSATION_REFERENCES = valid_refs
                print(f"Loaded {len(CONVERSATION_REFERENCES)} valid conversation references")
    except Exception as e:
        print(f"Error loading conversation references: {str(e)}")

def save_conversation_references():
    """Save conversation references to file."""
    try:
        with open(REFS_FILE, 'w') as f:
            data = {
                key: {
                    'activity_id': ref.activity_id,
                    'user': ref.user.as_dict() if ref.user else None,
                    'bot': ref.bot.as_dict() if ref.bot else None,
                    'conversation': ref.conversation.as_dict() if ref.conversation else None,
                    'channel_id': ref.channel_id,
                    'service_url': ref.service_url
                } for key, ref in CONVERSATION_REFERENCES.items()
            }
            json.dump(data, f, indent=2)
            print(f"Saved {len(CONVERSATION_REFERENCES)} conversation references")
    except Exception as e:
        print(f"Error saving conversation references: {str(e)}")

load_conversation_references()

@dataclass
class FastEmbedEmbedder:
    id: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384

    def get_embedding(self, text: str) -> List[float]:
        model = TextEmbedding(model_name=self.id)
        embeddings = model.embed(text)
        embedding_list = list(embeddings)[0]
        return embedding_list

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        embedding = self.get_embedding(text=text)
        return embedding, None

knowledge_base = AgentKnowledge(
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="email_memory",
        embedder=FastEmbedEmbedder(id="BAAI/bge-small-en-v1.5")
    )
)

knowledge_agent = Agent(
    name="Knowledge Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are an expert in looking for answers in the knowledge base.",
    knowledge=knowledge_base,
    search_knowledge=True,
    instructions=["Always look for answers in the knowledge base.", "If you don't find an answer, say 'No relevant information found'."],
    show_tool_calls=False,
    markdown=False
)

greeting_agent = Agent(
    name="Greeting Agent",
    description="You are an expert in conversational responses, acting like a human colleague.",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Respond as if you are a human colleague and keep responses friendly and professional.",
                  "Deflect politely to personal questions."],
    show_tool_calls=False,
    markdown=False
)

supervisor_team = Team(
    name="Supervisor Team",
    mode="route",
    members=[knowledge_agent, greeting_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are a supervisor who can analyze the query and route to the appropriate agent.",
    instructions=[
        "For greetings or personal questions, route to Greeting Agent.",
        "For rest of the questions, route to Knowledge Agent."
    ],
    show_tool_calls=False,
    markdown=False
)

async def send_adaptive_card():
    """Send an Adaptive Card to all stored conversations."""
    if not CONVERSATION_REFERENCES:
        print("No conversation references available.")
        return {"status": "error", "message": "No conversation references available. Interact with the bot first."}

    results = []
    for key, ref in CONVERSATION_REFERENCES.items():
        if not ref.conversation or not ref.conversation.id:
            print(f"Skipping invalid conversation reference {key}: missing conversation.id")
            results.append({"conversation": key, "status": "error", "message": "Invalid conversation.id"})
            continue
        print(f"Sending Adaptive Card to conversation: {key}")
        activity = Activity(
            type=ActivityTypes.message,
            channel_id=ref.channel_id,
            service_url=ref.service_url,
            conversation=ref.conversation,
            from_property=ref.bot,
            recipient=ref.user,
            attachments=[
                Attachment(
                    content_type="application/vnd.microsoft.card.adaptive",
                    content=ADAPTIVE_CARD
                )
            ]
        )
        try:
            await adapter.continue_conversation(
                ref,
                lambda turn_context: turn_context.send_activity(activity),
                APP_ID
            )
            print(f"Adaptive Card sent successfully to {key}")
            results.append({"conversation": key, "status": "success"})
        except Exception as e:
            print(f"Error sending Adaptive Card to {key}: {str(e)}")
            results.append({"conversation": key, "status": "error", "message": str(e)})

    return {"status": "completed", "results": results}

async def on_turn(turn_context: TurnContext):
    activity = turn_context.activity
    if not activity.conversation or not activity.conversation.id:
        print(f"Skipping invalid activity: channel_id={activity.channel_id}, no conversation.id")
        return

    global CONVERSATION_REFERENCES
    conversation_key = f"{activity.channel_id}:{activity.conversation.id}"
    if activity.channel_id == "msteams":
        if conversation_key not in CONVERSATION_REFERENCES or CONVERSATION_REFERENCES[conversation_key].service_url != activity.service_url:
            CONVERSATION_REFERENCES[conversation_key] = ConversationReference(
                activity_id=activity.id,
                user=activity.from_property,
                bot=activity.recipient,
                conversation=activity.conversation,
                channel_id=activity.channel_id,
                service_url=activity.service_url
            )
            print(f"Stored/Updated conversation reference for {conversation_key}")
            save_conversation_references()

    if activity.type == ActivityTypes.message:
        user_text = activity.text
        response_text = supervisor_team.run(user_text)
        response_text = response_text.content
        await turn_context.send_activity(MessageFactory.text(response_text))
        return

    if activity.type == ActivityTypes.conversation_update and activity.members_added:
        for member in activity.members_added:
            if member.id != activity.recipient.id:
                await turn_context.send_activity("Hello! How can I assist you today?")
                return

@app.get("/")
async def root():
    return {"message": "Teams Chatbot API is running"}

@app.post("/api/messages")
async def messages(request: Request):
    try:
        body = await request.json()
        activity = Activity().deserialize(body)
        auth_header = request.headers.get("Authorization", "")
        await adapter.process_activity(activity, auth_header, on_turn)
        return {}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/send-card")
async def send_card():
    """Endpoint to trigger broadcasting an Adaptive Card."""
    result = await send_adaptive_card()
    return result

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)