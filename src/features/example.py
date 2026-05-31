import time
from datetime import datetime
import paho.mqtt.client as mqtt

PROJECT_ID = "iot_09"  # previously could have been iot_07

class MqttApp:
    """
    Simple MQTT utility that can connect, subscribe, publish, and show status.
    """

    def __init__(self):
        # Create client with a clearer application name
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, f"{PROJECT_ID}_FullClient")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        # Runtime state
        self.is_connected = False
        self.topic_qos_map = {}     # {topic: qos}
        self.inbox = []             # list of dicts with timestamp/topic/message/qos

        # Broker details (adjust if needed)
        self.broker_host = "test.mosquitto.org"
        self.broker_port = 1883

    # ---------------- MQTT callbacks ----------------

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            print("[OK] Linked up with broker.")
        else:
            self.is_connected = False
            print(f"[ERR] Broker connection failed (rc={rc}).")

    def _on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        print("[INFO] Disconnected from broker.")

    def _on_message(self, client, userdata, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = msg.payload.decode("utf-8", errors="replace")
        rec = {"timestamp": ts, "topic": msg.topic, "message": payload, "qos": msg.qos}
        self.inbox.append(rec)
        print(f"[MSG] {ts} | {msg.topic} (QoS {msg.qos}) -> {payload}")

    # ---------------- Convenience methods ----------------

    def connect(self, keepalive=60):
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive)
            self.client.loop_start()
            time.sleep(1.5)  # brief wait for connack
            return self.is_connected
        except Exception as exc:
            print(f"[ERR] Exception while connecting: {exc}")
            return False

    def disconnect(self):
        if self.is_connected:
            self.client.loop_stop()
            self.client.disconnect()

    def subscribe(self, topic: str, qos: int = 0):
        if not self.is_connected:
            print("[WARN] Connect first.")
            return False
        result, _ = self.client.subscribe(topic, qos)
        if result == mqtt.MQTT_ERR_SUCCESS:
            self.topic_qos_map[topic] = qos
            print(f"[OK] Subscribed to '{topic}' with QoS {qos}.")
            return True
        print("[ERR] Subscription failed.")
        return False

    def publish(self, topic: str, message: str, qos: int = 0):
        if not self.is_connected:
            print("[WARN] Connect first.")
            return False
        result = self.client.publish(topic, message, qos=qos)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"[PUB] '{topic}' <= {message} (QoS {qos})")
            return True
        print("[ERR] Publish failed.")
        return False

    def status(self):
        print("===== MQTT STATUS =====")
        print(f"Connected      : {'Yes' if self.is_connected else 'No'}")
        print(f"Subscriptions  : {len(self.topic_qos_map)}")
        for t, q in self.topic_qos_map.items():
            print(f"  - {t} (QoS {q})")
        print(f"Messages seen  : {len(self.inbox)}")
        print("=======================")

    # ---------------- Menu loop ----------------

    def run_menu(self):
        print(f"{PROJECT_ID.upper()} :: Full MQTT Client")
        print("=" * 34)
        while True:
            print("\nMenu")
            print("  1) Connect to broker")
            print("  2) Subscribe to topic")
            print("  3) Publish message")
            print("  4) Show status")
            print("  5) Exit")
            choice = input("Select (1-5): ").strip()

            if choice == "1":
                print("[INFO] Connecting ...")
                ok = self.connect()
                print("[OK] Connected." if ok else "[ERR] Connection failed.")

            elif choice == "2":
                if not self.is_connected:
                    print("[WARN] Please connect first.")
                    continue
                topic = input("Topic: ").strip()
                if topic:
                    raw = input("QoS (0/1/2) [0]: ").strip()
                    qos = int(raw) if raw in ("0", "1", "2") else 0
                    self.subscribe(topic, qos)

            elif choice == "3":
                if not self.is_connected:
                    print("[WARN] Please connect first.")
                    continue
                topic = input("Topic: ").strip()
                payload = input("Message: ").strip()
                if topic and payload:
                    raw = input("QoS (0/1/2) [0]: ").strip()
                    qos = int(raw) if raw in ("0", "1", "2") else 0
                    self.publish(topic, payload, qos)

            elif choice == "4":
                self.status()

            elif choice == "5":
                print("[INFO] Shutting down ...")
                self.disconnect()
                break

            else:
                print("[WARN] Invalid option.")

def main():
    app = MqttApp()
    try:
        app.run_menu()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        app.disconnect()

if __name__ == "__main__":
    main()
