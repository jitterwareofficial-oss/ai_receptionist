import os
import json
import asyncio
import base64
import httpx
from fastapi import FastAPI, WebSocket, Response
from anthropic import AsyncAnthropic
from cartesia import Cartesia
from deepgram import AsyncDeepgramClient
from deepgram import (
    DeepgramClient,
    LiveOptions,
    LiveEvents
)
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gemma AI Receptionist - Claude Tool Calling & GHL Engine")

# Credential Loading from Render
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")

anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)
deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)

SYSTEM_PROMPT = """
You are "Gemma", a warm, concise AI front-desk receptionist for "Precision Auto Repair".
Your main goal is to collect the caller's First Name, Last Name, Phone Number, and Vehicle Issue.

CRITICAL VOICE RULES:
1. Keep every response under 20 words.
2. Never use markdown, asterisks, bold text, or lists.
3. Once you have successfully gathered all 4 required fields, immediately trigger the 'create_ghl_contact' tool.
"""

# 1. Claude Tool Definitions
GHL_TOOL_SCHEMA = {
    "name": "create_ghl_contact",
    "description": "Creates or updates a customer contact card in GoHighLevel with their vehicle details.",
    "input_schema": {
        "type": "object",
        "properties": {
            "first_name": {"type": "string", "description": "The customer's first name."},
            "last_name": {"type": "string", "description": "The customer's last name."},
            "phone": {"type": "string", "description": "The customer's phone number."},
            "issue": {"type": "string", "description": "Description of what is wrong with the vehicle."}
        },
        "required": ["first_name", "last_name", "phone", "issue"]
    }
}

# 2. GoHighLevel V2 API Integration
async def execute_ghl_contact_creation(first_name: str, last_name: str, phone: str, issue: str):
    url = "https://leadconnectorhq.com"
    headers = {
        "Authorization": f"Bearer {GHL_API_KEY}",
        "Version": "2021-04-15",
        "Content-Type": "application/json"
    }
    payload = {
        "locationId": GHL_LOCATION_ID,
        "firstName": first_name,
        "lastName": last_name,
        "phone": phone,
        "tags": ["AI Voice Lead"],
        "customFields": [
            {"id": "vehicle_issue", "value": issue}
        ]
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            print(f"[GHL CRM] API Response Status: {response.status_code}")
            return True
        except Exception as e:
            print(f"[GHL CRM Fail] Error executing tool: {e}")
            return False

@app.post("/twilio/voice")
async def handle_inbound_call():
    host_url = os.getenv("DEPLOYMENT_HOST", "://onrender.com")
    protocol = "wss" if "localhost" not in host_url else "ws"
    
    return Response(
        content=f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="{protocol}://{host_url}/media-stream" /></Connect></Response>',
        media_type="application/xml"
    )

@app.websocket("/media-stream")
async def handle_voice_gateway(twilio_ws: WebSocket):
    await twilio_ws.accept()
    print("[Pipeline Init] Twilio connected. Opening Deepgram Listener.")

    session = {"stream_sid": None, "history": []}
    tts_ctx = cartesia_client.tts.websocket()

    dg_options = LiveOptions(
        model="flux-general-en",
        language="en-US",
        encoding="mulaw",
        sample_rate=8000,
        interim_results=False
    )

    try:
        async with deepgram_client.listen.v1.connect(dg_options) as dg_ws:
            
            async def on_message(self, result, **kwargs):
                sentence = result.channel.alternatives[0].transcript
                if result.is_final and sentence.strip():
                    print(f"[Deepgram STT] User said: '{sentence}'")
                    asyncio.create_task(run_cognitive_response(sentence, twilio_ws, session, tts_ctx))

            dg_ws.on(LiveEvents.Transcript, on_message)

            while True:
                raw_frame = await twilio_ws.receive_text()
                packet = json.loads(raw_frame)

                if packet.get("event") == "start":
                    session["stream_sid"] = packet["start"]["streamSid"]
                    print(f"[Telephony] Live Stream ID: {session['stream_sid']}")
                    asyncio.create_task(run_cognitive_response("Hello", twilio_ws, session, tts_ctx))

                elif packet.get("event") == "media":
                    payload_b64 = packet["media"]["payload"]
                    await dg_ws.send(base64.b64decode(payload_b64))

                elif packet.get("event") == "stop":
                    break

    except Exception as e:
        print(f"[Pipeline Exception] Loop Runtime Error: {e}")
    finally:
        await twilio_ws.close()

# 3. Cognitive Response with Tool Execution Loop
async def run_cognitive_response(user_input: str, twilio_ws: WebSocket, session: dict, tts_ctx):
    session["history"].append({"role": "user", "content": user_input})

    try:
        # Call Claude 5.5 and pass the GHL Tool Schema explicitly
        response = await anthropic_client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=session["history"],
            tools=[GHL_TOOL_SCHEMA]
        )
        
        ai_text = ""
        tool_call_detected = False
        tool_use_id = None
        tool_inputs = {}

        # Parse Claude's content blocks for text responses or tool triggers
        for block in response.content:
            if block.type == "text":
                ai_text += block.text
            elif block.type == "tool_use":
                tool_call_detected = True
                tool_use_id = block.id
                tool_inputs = block.input

        # Handle Tool Call execution if Claude decides to use the tool
        if tool_call_detected:
            print(f"[Claude Tool Trigger] Executing GHL Sync: {tool_inputs}")
            success = await execute_ghl_contact_creation(
                first_name=tool_inputs.get("first_name", "Unknown"),
                last_name=tool_inputs.get("last_name", "Caller"),
                phone=tool_inputs.get("phone", ""),
                issue=tool_inputs.get("issue", "General Repair Inquiry")
            )
            
            # Send the confirmation block back to Claude's memory loop
            session["history"].append({
                "role": "assistant",
                "content": response.content
            })
            session["history"].append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "Success: Contact logged inside GHL CRM." if success else "Error: GHL API rejected payload."
                    }
                ]
            })
            
            # Request final sentence from Claude confirming registration
            follow_up = await anthropic_client.messages.create(
                model="claude-haiku-5-5",
                max_tokens=128,
                system=SYSTEM_PROMPT,
                messages=session["history"]
            )
            ai_text = follow_up.content[0].text

        if ai_text.strip():
            print(f"[Claude 5.5] Response: '{ai_text}'")
            session["history"].append({"role": "assistant", "content": ai_text})

            # Stream audio out over Cartesia Sonic
            output_format = {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000}
            response_stream = tts_ctx.send(
                model_id="sonic-latest",
                transcript=ai_text,
                voice_id="62ae83ad-4f6a-430b-af41-a9bede9286ca",
                output_format=output_format,
                stream=True
            )

            for chunk in response_stream:
                b64_audio = base64.b64encode(chunk["audio"]).decode("utf-8")
                await twilio_ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": session["stream_sid"],
                    "media": {"payload": b64_audio}
                }))
                    
    except Exception as exc:
        print(f"[Cognitive Error] Processing Failure: {exc}")
