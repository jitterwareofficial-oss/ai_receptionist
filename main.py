import os
import json
import asyncio
import base64
import httpx
from fastapi import FastAPI, WebSocket, Response
from anthropic import AsyncAnthropic
from cartesia import Cartesia
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gemma AI Receptionist - Cartesia Unified Stream")

# Credentials loaded directly from Render
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")

anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

SYSTEM_PROMPT = """
You are "Gemma", a warm, concise AI front-desk receptionist for "Precision Auto Repair".
Your main goal is to collect the caller's First Name, Last Name, Phone Number, and Vehicle Issue.

CRITICAL VOICE RULES:
1. Keep every response under 20 words.
2. Never use markdown, asterisks, bold text, or lists.
3. Once you have successfully gathered all 4 fields, tell the user you are logging it.
"""

# GoHighLevel V2 API Call
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
        "tags": ["AI Voice Lead"]
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            print(f"[GHL CRM] Contact Creation Status: {response.status_code}")
            return True
        except Exception as e:
            print(f"[GHL CRM Fail] Error: {e}")
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
    print("[Pipeline Init] Connected to Twilio. Opening Cartesia Loops.")

    session = {"stream_sid": None, "history": []}
    
    # Open both text-to-speech and speech-to-text contexts via Cartesia's official SDK
    tts_ctx = cartesia_client.tts.websocket()
    
    try:
        # Open Cartesia Transcription Context using their standard web socket client interface
        async with cartesia_client.stt.websocket() as stt_ws:
            
            # Configure transcription parameters
            await stt_ws.send(json.dumps({
                "setup": {
                    "model_id": "ink-2",
                    "language": "en",
                    "audio": {
                        "encoding": "mulaw",
                        "sample_rate": 8000
                    }
                }
            }))

            async def listen_to_transcripts():
                async for message in stt_ws:
                    resp = json.loads(message)
                    if resp.get("type") == "transcript" and resp.get("final"):
                        user_text = resp["transcript"]["text"]
                        if user_text.strip():
                            print(f"[Cartesia STT] User said: '{user_text}'")
                            
                            # Simple extraction trigger for GoHighLevel integration testing
                            if "windshield" in user_text.lower() or "fix" in user_text.lower():
                                asyncio.create_task(execute_ghl_contact_creation("Voice", "Lead", "8194141677", user_text))
                            
                            asyncio.create_task(run_cognitive_response(user_text, twilio_ws, session, tts_ctx))

            asyncio.create_task(listen_to_transcripts())

            while True:
                raw_frame = await twilio_ws.receive_text()
                packet = json.loads(raw_frame)

                if packet.get("event") == "start":
                    session["stream_sid"] = packet["start"]["streamSid"]
                    print(f"[Telephony] Live Stream ID: {session['stream_sid']}")
                    asyncio.create_task(run_cognitive_response("Hello", twilio_ws, session, tts_ctx))

                elif packet.get("event") == "media":
                    payload_b64 = packet["media"]["payload"]
                    # Stream raw telephone audio chunks directly up to Cartesia transcription
                    await stt_ws.send(json.dumps({"audio": payload_b64}))

                elif packet.get("event") == "stop":
                    break

    except Exception as e:
        print(f"[Pipeline Exception] Loop Error: {e}")
    finally:
        await twilio_ws.close()

async def run_cognitive_response(user_input: str, twilio_ws: WebSocket, session: dict, tts_ctx):
    session["history"].append({"role": "user", "content": user_input})

    try:
        response = await anthropic_client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=session["history"]
        )
        
        ai_text = response.content[0].text
        print(f"[Claude 5.5] Response: '{ai_text}'")
        session["history"].append({"role": "assistant", "content": ai_text})

        output_format = {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000}
        response_stream = tts_ctx.send(
            model_id="sonic-latest",
            transcript=ai_text,
            voice_id="79a125e8-cd45-4c13-8a67-ac375d374e5d",
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
