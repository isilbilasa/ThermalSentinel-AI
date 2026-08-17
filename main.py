import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import paho.mqtt.client as mqtt

app = FastAPI(title="Thermal Edge-to-Cloud API")

# 1. CORS izinleri (Tarayıcının WebSocket'i engellemesini önler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. WebSocket bağlantı yöneticisi
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Yeni istemci bağlandı! Aktif bağlantı sayısı: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print("İstemci ayrıldı.")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Gönderilemedi: {e}")

manager = ConnectionManager()
main_loop = None  # FastAPI'nin ana async döngüsü

# 3. MQTT mesaj yakalayıcı 
def on_mqtt_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        print(f" Kare ID: {payload.get('frame_id')} | İnsan: {payload.get('total_humans')} | Anomali: {payload.get('anomaly_count')}")
        
        # MQTT mesajını FastAPI'nin ana asyncio döngüsüne güvenli şekilde ekliyoruz
        if main_loop and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(manager.broadcast(payload), main_loop)
    except Exception as e:
        print(f"[MQTT okuma hatası] {e}")

# 4. Sunucu başlarken MQTT istemcisini arka plan thread'inde çalıştır
@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    
    client = mqtt.Client()
    client.on_message = on_mqtt_message
    
    try:
        client.connect("broker.emqx.io", 1883, 60)
        client.subscribe("thermal/sentinel/alerts")
        client.loop_start()  # Arka planda kesintisiz dinleme başlatır
        print(" Broker'a bağlandı. Topic: 'thermal/sentinel/alerts' dinleniyor...")
    except Exception as e:
        print(f"Bağlanılamadı: {e}")

# 5. Canlı webSocket endpoint'i
@app.websocket("/ws/live-telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Bağlantıyı canlı tutar
    except WebSocketDisconnect:
        manager.disconnect(websocket)