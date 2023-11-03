import paho.mqtt.client as mqtt
from time import sleep

publish_topic = "window/"
publish_subtopic = "open"
message = "This is a test"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: " + str(rc))
    print("Waiting for 2 seconds")
    sleep(2)

    print("Sending message.")
    client.publish(f"{publish_topic}{publish_subtopic}", f"{message}")

client = mqtt.Client()
client.on_connect = on_connect
client.connect("localhost", 1883, 60)
client.loop_forever()
