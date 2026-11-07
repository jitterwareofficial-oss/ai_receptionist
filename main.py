import base64
import os
import json
import asyncio
import httpx
from fastapi import FastAPI, WebSocket, Response
from anthropic import AsyncAnthropic
from cartesia import Cartesia
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Real-Time Conversational AI Voice Gateway")

# Setup clean environment variables
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")

anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

SYSTEM_PROMPT = """
You are "Gemma", a warm, concise AI front-desk receptionist for "Precision Auto Repair".
Your primary goal is to gather the user's name and details about their vehicle problem.

CRITICAL VOICE RULES:
1. Keep every response under 20 words. 
2. Never use markdown, asterisks, or bold text.
"""

# GoHighLevel V2 CRM Background Sync Integration
async def sync_lead_to_ghl_crm(conversation_history: list):
    """
    Analyzes the conversation safely in the background. If a vehicle issue is discussed,
    it automatically pushes a clean lead contact record to your GoHighLevel account.
    """
    if not GHL_API_KEY or not GHL_LOCATION_ID:
        return

    # Simple keyword heuristic check to prevent empty CRM spam cards
    history_string = json.dumps(conversation_history).lower()
    if not any(keyword in history_string for keyword in ["fix", "windshield", "repair", "car", "broken"]):
        return

    url = "https://leadconnectorhq.com"
    headers = {
        "Authorization": f"Bearer {GHL_API_KEY}",
        "Version": "2021-04-15",
        "Content-Type": "application/json"
    }
    
    # Extract structural conversation records for the CRM log entry
    payload = {
        "locationId": GHL_LOCATION_ID,
        "firstName": "Voice Line",
        "lastName": "Incoming Lead",
        "tags": ["AI Voice Call Lead"],
        "customFields": [
            {"id": "vehicle_issue", "value": f"Latest call logs context: {history_string[-300:]}"}
        ]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            print(f"[GHL CRM Background Sync] Status Code: {response.status_code}")
        except Exception as err:
            print(f"[GHL CRM Fail] Background reporting encountered a network error: {err}")

@app.post("/twilio/voice")
async def handle_inbound_call():
    host_url = os.getenv("DEPLOYMENT_HOST", "ai-receptionist-ydjs.onrender.com")
    protocol = "wss" if "localhost" not in host_url else "ws"
    
    twiml_instruction = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="{protocol}://{host_url}/media-stream" />
        </Connect>
    </Response>
    """
    return Response(content=twiml_instruction, media_type="application/xml")

@app.websocket("/media-stream")
async def handle_voice_gateway(twilio_ws: WebSocket):
    await twilio_ws.accept()
    print("[Pipeline Init] Twilio connected via WebSocket stream.")

    session = {"stream_sid": None, "history": []}

    try:
        ctx = cartesia_client.tts.websocket()

        while True:
            raw_frame = await twilio_ws.receive_text()
            packet = json.loads(raw_frame)

            if packet.get("event") == "start":
                session["stream_sid"] = packet["start"]["streamSid"]
                print(f"[Telephony] Live Stream ID: {session['stream_sid']}")
                asyncio.create_task(run_cognitive_response("Hello", twilio_ws, session, ctx))

            elif packet.get("event") == "media":
                pass

            elif packet.get("event") == "stop":
                print("[Telephony] Call ended by user. Triggering background CRM sync.")
                # Run the CRM integration loop right as the user hangs up to keep voice streaming clean
                asyncio.create_task(sync_lead_to_ghl_crm(session["history"]))
                break

    except Exception as e:
        print(f"[Pipeline Exception] Runtime error inside processing loop: {e}")
    finally:
        await twilio_ws.close()

async def run_cognitive_response(user_input: str, twilio_ws: WebSocket, session: dict, tts_ctx):
    session["history"].append({"role": "user", "content": user_input})

    try:
        # 1. Query Claude 5.5 Haiku
        response = await anthropic_client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=session["history"]
        )
        
        ai_text = response.content[0].text
        print(f"[Claude 5.5] Generated Response: '{ai_text}'")
        session["history"].append({"role": "assistant", "content": ai_text})

        # 2. Use Cartesia official SDK context to request raw mu-law streaming chunks safely
        output_format = {
            "container": "raw",
            "encoding": "pcm_mulaw",
            "sample_rate": 8000
        }
        
        response_stream = tts_ctx.send(
            model_id="sonic-latest",
            transcript=ai_text,
            voice_id="62ae83ad-4f6a-430b-af41-a9bede9286ca",
            output_format=output_format,
            stream=True
        )

        for chunk in response_stream:
            audio_bytes = chunk["audio"]
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

            await twilio_ws.send_text(json.dumps({
                "event": "media",
                "streamSid": session["stream_sid"],
                "media": {"payload": b64_audio}
            }))
                    
    except Exception as exc:
        print(f"[Cognitive Fail] Voice stream tracking encountered an error: {exc}")
