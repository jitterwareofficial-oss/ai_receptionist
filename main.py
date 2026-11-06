import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, Response
from anthropic import AsyncAnthropic
import httpx
from cartesia import Cartesia
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Pure Cartesia & Claude 5.5 Voice Gateway Engine")

# Setup clean, single-vendor developer authentications
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize the official Cartesia SDK instance
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

SYSTEM_PROMPT = """
You are "Sam", a warm, concise AI front-desk receptionist for "Precision Auto Repair".
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
    print("[Pipeline Init] Twilio connected. Opening unified Cartesia loops.")

    # Target the production Cartesia STT and TTS endpoint loops
    cartesia_stt_url = "wss://api.cartesia.ai/stt/turns/websocket"
    cartesia_tts_url = "wss://api.cartesia.ai/tts/websocket?v=2024-06-10"

    session = {"stream_sid": None, "history": []}

    # Import specialized client connector inside the thread routine
    from websockets.client import connect as ws_connect

    try:
        # Open persistent connections directly to Cartesia's STT (Ink-2) server
        async with ws_connect(f"{cartesia_stt_url}?api_key={CARTESIA_API_KEY}") as stt_ws:
            
            # Configure Cartesia Ink-2 for Twilio's standard raw telephony format
            stt_setup = {
                "model": "ink-2",
                "language": "en",
                "sample_rate": 8000,
                "encoding": "mulaw"
            }
            await stt_ws.send(json.dumps({"setup": stt_setup}))

            # Spin up a concurrent background background-worker task to continuously listen 
            # for text transcripts returning from Cartesia Ink-2
            async def listen_to_cartesia_transcripts():
                async for message in stt_ws:
                    resp = json.loads(message)
                    
                    # Cartesia Ink-2 emits distinct structured events when transcription completes
                    if resp.get("type") == "transcript" and resp.get("final"):
                        user_text = resp["transcript"]["text"]
                        if user_text.strip():
                            print(f"[Cartesia Ink-2] User Said: '{user_text}'")
                            # Process conversational text immediately with Claude 5.5
                            asyncio.create_task(
                                run_cognitive_response(user_text, twilio_ws, session, cartesia_tts_url)
                            )
            
            asyncio.create_task(listen_to_cartesia_transcripts())

            # Read raw frames arriving from Twilio
            while True:
                raw_frame = await twilio_ws.receive_text()
                packet = json.loads(raw_frame)

                if packet.get("event") == "start":
                    session["stream_sid"] = packet["start"]["streamSid"]
                    print(f"[Telephony] Live Stream ID: {session['stream_sid']}")
                    # Trigger an initial greeting hand-shake block
                    asyncio.create_task(run_cognitive_response("Hello", twilio_ws, session, cartesia_tts_url))

                elif packet.get("event") == "media":
                    # Send raw audio payload chunk directly down to Cartesia Ink-2
                    audio_payload = packet["media"]["payload"]
                    await stt_ws.send(json.dumps({"audio": audio_payload}))

                elif packet.get("event") == "stop":
                    break

    except Exception as e:
        print(f"[Pipeline Exception] Runtime error inside processing loop: {e}")
    finally:
        await twilio_ws.close()

# =====================================================================
# COGNITION TO COUPLING PATHWAY
# =====================================================================
async def run_cognitive_response(user_input: str, twilio_ws: WebSocket, session: dict, tts_url: str):
    session["history"].append({"role": "user", "content": user_input})
    from websockets.client import connect as ws_connect

    try:
        # Query Claude Haiku 5.5
        response = await anthropic_client.messages.create(
            model="claude-haiku-5.5",
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=session["history"]
        )
        
        ai_text = response.content[0].text
        print(f"[Claude 5.5] Generated Response: '{ai_text}'")
        session["history"].append({"role": "assistant", "content": ai_text})

        # Connect directly to Cartesia Sonic (TTS) to generate playback bytes
        async with ws_connect(f"{tts_url}&api_key={CARTESIA_API_KEY}") as tts_ws:
            tts_setup = {
                "context_id": session["stream_sid"],
                "model_id": "sonic-english",
                "embedding": {"style": "formal", "id": "e0b49cb3-cb9c-411a-9694-ca28389656fb"},
                "output_format": {"container": "raw", "encoding": "mulaw", "sample_rate": 8000}
            }
            await tts_ws.send(json.dumps({"setup": tts_setup, "input": {"transcript": ai_text, "continue": False}}))

            async for msg in tts_ws:
                audio_packet = json.loads(msg)
                if "audio" in audio_packet:
                    # Package bytes immediately back into Twilio format
                    await twilio_ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": session["stream_sid"],
                        "media": {"payload": audio_packet["audio"]}
                    }))
                    
    except Exception as exc:
        print(f"[Cognitive Fail] Generation tracking encountered an exception: {exc}")
