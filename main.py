import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

app = FastAPI()

# 設定允許前端跨域連線（讓 HTML 網頁可以順利跟 Python 通訊）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
  message: str


# 定義敏感資料遮蔽處理函式
def mask_sensitive_data(text: str):
  # 1. 台灣身分證字號遮蔽 (英文大寫 1 碼 + 1或2 + 8碼數字)
  id_pattern = r'[A-Z][12]\d{8}'
  text_masked = re.sub(id_pattern, '[ID_CARD]', text)

  # 2. 手機號碼遮蔽 (09開頭共10碼)
  phone_pattern = r'09\d{8}'
  text_masked = re.sub(phone_pattern, '[PHONE_NUMBER]', text_masked)

  return text_masked


@app.post("/chat")
def chat_with_ai(req: ChatRequest):
  original_message = req.message  # 取得使用者輸入的原句

  # 透過 Python 進行遮蔽，確保敏感個資留在本地伺服器、絕不外流給模型
  safe_message = mask_sensitive_data(original_message)

  # 透過 system prompt 來「訓練與引導」模型扮演專屬的金融助理
  response = ollama.chat(
      model="cwchang/llama3-taide-lx-8b-chat-alpha1",
      messages=[
          {
              "role": "system",
              "content": (
                  "你是一位專業、親切且嚴謹的金融理財助理（名為「金融小幫手」）。"
                  "請用繁體中文回覆使用者，專注於提供理財、ETF、股票、資產配置、存款規劃與總體經濟知識。"
                  "原則："
                  "1. 回答要條理分明、重點突顯。"
                  "2. 語氣溫和、具備引導性。"
                  "3. 務必在回答中或結尾加上提醒：內容僅供一般資訊與教育參考，不構成個人化投資建議。"
              ),
          },
          {
              "role": "user",
              "content": safe_message,
          },  # 實際送給模型的是遮蔽後的安全句子
      ],
  )

  # 同時回傳原句、遮蔽後句子與 AI 回覆，供前端比對展示
  return {
      "original": original_message,
      "masked": safe_message,
      "reply": response["message"]["content"],
  }