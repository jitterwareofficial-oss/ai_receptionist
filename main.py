import base64
import os
import json
import asyncio
import httpx
from fastapi import FastAPI, Request, Response
from anthropic import AsyncAnthropic
from cartesia import Cartesia
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gemma AI Receptionist - Stable Twilio Native Engine")

# Credentials loaded directly from Render
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")

anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

SYSTEM_PROMPT = """
You are "Gemma", a warm, concise AI front-desk receptionist for "Precision Auto Repair".
Your primary goal is to gather the caller's First Name, Last Name, Phone Number, and Vehicle Issue.

CRITICAL VOICE RULES:
1. Keep every response under 20 words.
2. Never use markdown, asterisks, bold text, or lists.
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
        "tags": ["AI Voice Call Lead"],
        "customFields": [{"id": "vehicle_issue", "value": issue}]
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            print(f"[GHL CRM] Contact Creation Status: {response.status_code}")
            return True
        except Exception as e:
            print(f"[GHL CRM Fail] Error: {e}")
            return False

# 1. Initial greeting when a customer calls your Twilio number
@app.post("/twilio/voice")
async def handle_inbound_call():
    initial_greeting = "Hello! Welcome to Precision Auto Repair. May I have your name, please?"
    
    # FIXED: Added the correct /v1/tts/bytes production path for Cartesia
    audio_url = f"https://cartesia.ai{httpx.URL(initial_greeting)}&voice_id=62ae83ad-4f6a-430b-af41-a9bede9286ca&output_format=wav"
    
    # Twilio plays the Cartesia audio, listens for the user's name, and sends it to /twilio/respond
    twiml_instruction = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Play>{audio_url}</Play>
        <Gather input="speech" action="/twilio/respond" speechTimeout="auto" />
    </Response>
    """
    return Response(content=twiml_instruction, media_type="application/xml")

# 2. Twilio automatically transcribes the user's spoken words and posts the text here
@app.post("/twilio/respond")
async def handle_user_response(request: Request):
    form_data = await request.form()
    user_speech = form_data.get("SpeechResult", "")
    caller_phone = form_data.get("From", "8194141677") # Grabs the actual caller's number automatically
    
    print(f"[Twilio Transcribed] User Said: '{user_speech}'")
    
    if not user_speech.strip():
        # Fallback if the user stayed completely silent
        silent_audio = "https://cartesia.aiI%20didn't%20catch%20that.%20Could%20you%20repeat%20it?&voice_id=62ae83ad-4f6a-430b-af41-a9bede9286ca&output_format=wav"
        return Response(content=f"<Response><Play>{silent_audio}</Play><Gather input='speech' action='/twilio/respond'/></Response>", media_type="application/xml")

    # If the user mentions a car issue or windshield, push it straight to GoHighLevel in the background
    if any(k in user_speech.lower() for k in ["windshield", "fix", "repair", "broken", "glass"]):
        asyncio.create_task(execute_ghl_contact_creation("Voice", "Lead", caller_phone, user_speech))

    # Send the clean transcribed text straight to Claude 5.5
    response = await anthropic_client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_speech}]
    )
    
    ai_text = response.content[0].text if isinstance(response.content, list) else response.content.text
    print(f"[Claude 5.5] Response: '{ai_text}'")
    
    # FIXED: Added the correct /v1/tts/bytes production path for Cartesia
    reply_audio_url = f"https://cartesia.ai{httpx.URL(ai_text)}&voice_id=62ae83ad-4f6a-430b-af41-a9bede9286ca&output_format=wav"
    
    # Loop back to continue the conversation seamlessly
    twiml_instruction = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Play>{reply_audio_url}</Play>
        <Gather input="speech" action="/twilio/respond" speechTimeout="auto" />
    </Response>
    """
    return Response(content=twiml_instruction, media_type="application/xml")
