#!/usr/bin/env python3
"""
Telegram Image Sender
Sends images via Telegram Bot API (not from personal account)
"""
import os
import sys
import requests

def send_photo_via_bot(bot_token, chat_id, image_path, caption=None):
    """
    Send a photo to Telegram chat using Bot API
    
    Args:
        bot_token: Telegram bot token (from env var or parameter)
        chat_id: Target chat ID
        image_path: Path to image file
        caption: Optional caption text
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    # Send the photo
    with open(image_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': chat_id}
        if caption:
            data['caption'] = caption
        
        response = requests.post(url, files=files, data=data)
        return response.json()

if __name__ == '__main__':
    # Get bot token from env
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token and len(sys.argv) > 1:
        bot_token = sys.argv[1]
    
    if not bot_token:
        print("Error: TELEGRAM_BOT_TOKEN env var not set and no token provided")
        print("Usage: python telegram_send_image.py [bot_token] [image_path] [chat_id] [caption]")
        sys.exit(1)
    
    # Get chat ID from env or arg
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if len(sys.argv) > 3:
        chat_id = sys.argv[3]
    
    if not chat_id:
        print("Error: TELEGRAM_CHAT_ID env var not set and no chat_id provided")
        print("Usage: python telegram_send_image.py [bot_token] [image_path] [chat_id] [caption]")
        sys.exit(1)
    
    # Get image path
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    if not image_path:
        print("Error: No image path provided")
        print("Usage: python telegram_send_image.py [bot_token] [image_path] [chat_id] [caption]")
        sys.exit(1)
    
    # Get optional caption
    caption = sys.argv[4] if len(sys.argv) > 4 else None
    
    print(f"Sending {image_path} to chat {chat_id}...")
    result = send_photo_via_bot(bot_token, chat_id, image_path, caption)
    
    if result.get('ok'):
        print("Image sent successfully!")
        print(f"Message ID: {result['result']['message_id']}")
    else:
        print(f"Error: {result}")
