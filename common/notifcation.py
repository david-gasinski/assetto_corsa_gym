import os
import requests

class NotifcationClient():
    
    def __init__(self, api_key: str = None, user_key: str = None) -> None:
        self.client = None
        self.api_key = api_key
        self.user_key = user_key
                
    def send_notifcation(self, msg: str, title: str, image: str = None) -> None:
        """
            Sends a notification to the configured users in self.user_key
        """
        if self.api_key is None or self.user_key is None:
            return
        
        if image is not None:
            # check image exists
            if not os.path.exists(os.path.abspath(image)):
                return
            
            attachment = {
                "attachment" : (image, open(image, "rb"), "image/jpeg")
            }
                    
        requests.post("https://api.pushover.net/1/messages.json", data= {
            "token": self.api_key,
            "user": self.user_key,
            "message": msg,
            "title": title
        }, files= attachment if image else None)
        