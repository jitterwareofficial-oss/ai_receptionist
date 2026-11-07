import base64
import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, Response
from anthropic import AsyncAnthropic
from cartesia import Cartesia
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="Real-Time Conversational AI Voice Gateway")

# Setup clean environment variables
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize the official Cartesia SDK instance
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

SYSTEM_PROMPT = """
You are "Gemma", a warm, concise AI front-desk receptionist for "Precision Auto Repair".
CRITICAL: Keep every response under 20 words. Never use markdown, asterisks, or bold text.
"""

@app.post("/twilio/voice")
async def handle_inbound_call():
    host_url = os.getenv("DEPLOYMENT_HOST", "://onrender.com")
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
        # We let the Cartesia Python SDK manage the TTS WebSocket loop for us automatically!
        # This completely bypasses the manually written ws_connect handshake errors
        ctx = cartesia_client.tts.websocket()

        # Read raw frames arriving from Twilio
        while True:
            raw_frame = await twilio_ws.receive_text()
            packet = json.loads(raw_frame)

            if packet.get("event") == "start":
                session["stream_sid"] = packet["start"]["streamSid"]
                print(f"[Telephony] Live Stream ID: {session['stream_sid']}")
                # Trigger Gemma's initial greeting
                asyncio.create_task(run_cognitive_response("Hello", twilio_ws, session, ctx))

            elif packet.get("event") == "media":
                # Twilio's incoming caller audio bytes can go straight to your backend logic
                # For basic setup, we focus on responding dynamically via text-to-speech chunks
                pass

            elif packet.get("event") == "stop":
                print("[Telephony] Call ended by user.")
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
        
        # Streams bytes down directly using Cartesia's official socket handler
        response_stream = tts_ctx.send(
            model_id="sonic-latest",
            transcript=ai_text,
            voice_id="62ae83ad-4f6a-430b-af41-a9bede9286ca", # "Doctor" - Clear, professional voice
            output_format=output_format,
            stream=True
        )

        for chunk in response_stream:
            # Convert raw audio bytes into a base64 text string for Twilio
            audio_bytes = chunk["audio"]
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

            # Package the string safely into Twilio's JSON envelope
            await twilio_ws.send_text(json.dumps({
                "event": "media",
                "streamSid": session["stream_sid"],
                "media": {"payload": b64_audio}
            }))
                    
    except Exception as exc:
        print(f"[Cognitive Fail] Voice stream tracking encountered an error: {exc}")