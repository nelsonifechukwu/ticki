import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()
# For Development Alone!!!!!!!!!!!!!!!!
class Config:
    POSTGRES_USER = os.environ.get('POSTGRES_USER')
    POSTGRES_PASS = os.environ.get('POSTGRES_PASS')
    POSTGRES_URL = os.environ.get('POSTGRES_URL')
    SECRET_KEY = os.environ.get('SECRET_KEY')

    CLIENT_ID = os.environ.get("CLIENT_ID")
    CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
    EXTS = set(['png', 'jpg', 'jpeg'])
    SQLALCHEMY_DATABASE_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_URL}"
    # SQLALCHEMY_DATABASE_URI ="sqlite:///test"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CACHE_TYPE = "SimpleCache"
    GOOGLE_DISCOVERY_URL = (
    "https://accounts.google.com/.well-known/openid-configuration"
)


# class Mail:
#   def __init__(self, name, mail):
#     self.name = name
#     self.mail = mail
#     self.all = self.name + self.mail

#   def show(self):
#     print(self.name + " " + self.mail + " " + self.all)


# c = Mail("Nelson", "nelson.elijah@yahoo.com")
# c.name = "Blessing"
# c.mail = "b@yahoo.com"
# c.show()
# print(c.all)
