# python3.6

from sentiment import *

from paho.mqtt import client as mqtt_client
import urllib.request

# CONFIG
# broker := Rhasspy IP
host = 'localhost'
port_mqtt = 12183
port_api = 12101

# Forwarding to UX/CS
uxcs_ip = '123.456.789'

# MQTT TOPICS
MQTT_topicIntent = "hermes/intent/#"

# Insert a creative name
FILE_NAME = "lastIntent.mp3"

# generate client ID with pub prefix randomly
client_id = "SentimentAnalysis"

# username = 'user'
# password = '12345678'

emotionAnalyser = EmotionAnalyser(categorial_output=True, show_confidence=True)


def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(host, port_mqtt)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):

		# Get topic from mqtt message
        topic = str(msg.topic)
        topic = topic.strip()

		# Handle mqtt message
        if "hermes/intent/" in topic:
            downloadLastWav()
            handle()
        else:
            # Does not matter...
            None

    # Subscripe to topics
    client.subscribe(MQTT_topicIntent)

    client.on_message = on_message
    print("Waiting for Intents...")


def downloadLastWav():
    url = "http://" + host + ":" + str(port_api) + "/api/play-recording"
    urllib.request.urlretrieve(url, FILE_NAME)


def handle():
    file = FILE_NAME
    emotion = emotionAnalyser.predict(file = file)
    print(emotion)


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


def sendResult(emotion):
    json_msg = {
    "entities": [],
    "intent": {
        "confidence": 0,
        "name": ""
    },
    "emotion": {
        "name": emotion.get('emotion', 'neutral'),
        "confidence": emotion.get('confidence', 0)
    },
    "raw_text": "",
    "raw_tokens": [
        ""
    ],
    "recognize_seconds": 0,
    "slots": {},
    "speech_confidence": 1,
    "text": "",
    "tokens": [
        ""
    ],
    "wakeword_id": None
    }

    # req =  request.Request(uxcs_ip, data=json_msg) # this will make the method "POST"


if __name__ == '__main__':
    run()
