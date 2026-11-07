import os
import json
from fastapi import FastAPI, Request, Response
from anthropic import AsyncAnthropic
from cartesia import Cartesia
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Twilio + Claude 5.5 + Cartesia Audio Answering Service")

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

SYSTEM_PROMPT = """
You are "Gemma", a warm, concise AI front-desk receptionist for "Precision Auto Repair".
CRITICAL: Keep every response under 20 words. Never use markdown, asterisks, or bold text.
"""

# 1. Initial Greeting when someone calls your Twilio number
@app.post("/twilio/voice")
async def handle_inbound_call():
    initial_greeting = "Hello! Welcome to Precision Auto Repair. How can I help you today?"
    
    # We use Cartesia's local Python SDK helper to quickly fetch standard audio playback bytes
    audio_url = f"https://cartesia.ai{initial_greeting}&voice_id=e0b49cb3-cb9c-411a-9694-ca28389656fb"
    
    # Twilio plays your Cartesia greeting, then listens for the user's spoken reply
    twiml_instruction = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Play>{audio_url}</Play>
        <Gather input="speech" action="/twilio/respond" speechTimeout="auto" hints="appointment, repair, oil change">
            <Play>https://cartesia.ai</Play>
        </Gather>
    </Response>
    """
    return Response(content=twiml_instruction, media_type="application/xml")

# 2. Twilio automatically captures the caller's spoken words, transcribes it, and POSTs it here!
@app.post("/twilio/respond")
async def handle_user_response(request: Request):
    form_data = await request.form()
    user_speech = form_data.get("SpeechResult", "")
    
    print(f"[Twilio Transcribed] User Said: {user_speech}")
    
    if not user_speech:
        # If they remained silent, prompt them again gently
        silent_audio = "https://cartesia.aiI%20didn't%20catch%20that.%20Could%20you%20repeat%20it?&voice_id=e0b49cb3-cb9c-411a-9694-ca28389656fb"
        return Response(content=f"<Response><Play>{silent_audio}</Play><Gather input='speech' action='/twilio/respond'/></Response>", media_type="application/xml")

    # Pass the clean text right to Claude 5.5
    response = await anthropic_client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_speech}]
    )
    
    ai_text = response.content[0].text
    print(f"[Claude 5.5] Response: {ai_text}")
    
    # Convert Claude's text into beautiful Cartesia speech audio link instantly
    reply_audio_url = f"https://cartesia.ai{ai_text}&voice_id=e0b49cb3-cb9c-411a-9694-ca28389656fb"
    
    # Loop back to continue the live voice conversation cleanly
    twiml_instruction = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Play>{reply_audio_url}</Play>
        <Gather input="speech" action="/twilio/respond" speechTimeout="auto" />
    </Response>
    """
    return Response(content=twiml_instruction, media_type="application/xml")
