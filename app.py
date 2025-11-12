from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Literal
from datetime import datetime
from uuid import uuid4
import pickle, os, json, requests, re, string, numpy as np
import fitz
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configuração do banco SQLite
DATABASE_URL = "sqlite:///./history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo da tabela
class History(Base):
    __tablename__ = "history"
    id = Column(String, primary_key=True, index=True)
    created_at = Column(String)
    user_id = Column(String)
    input = Column(Text)
    output = Column(Text)
    chosen_version = Column(String, nullable=True)
    feedback_reason = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

# Inicialização do FastAPI
app = FastAPI(title="CRM Inteligente Globo", description="Assistente estratégico para campanhas digitais")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class ChatInput(BaseModel):
    text: str
    platform: str
    content_type: str
    user_id: str

class ChatResponse(BaseModel):
    suggestion_id: str
    chosen_version: str
    user_id: str

class ChatFeedback(BaseModel):
    suggestion_id: str
    user_id: str
    feedback: Literal[
        "Boas sugestões",
        "Boa explicação",
        "Sugestões genéricas",
        "Não condiz com o tema",
        "Texto fraco",
        "Formato inadequado",
        "Horário ruim",
        "Hashtags irrelevantes"
    ]

# Carregar modelos ML
models_dir = "models"
with open(os.path.join(models_dir, "modelo_xgb.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(models_dir, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# Funções auxiliares
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@\w+|#\w+|https?://\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

keywords = ["humor", "música", "moda", "memes", "sustentabilidade", "diversidade", "games", "tecnologia", "lifestyle",
            "novela", "jornalismo", "esporte", "reality", "entretenimento", "cultura", "celebridade", "globoplay", "tvglobo"]
platform_scores = {"instagram": 0.3, "tiktok": 0.4, "youtube": 0.2, "twitter": 0.1}

def get_best_time(platform, content_type):
    p, c = platform.lower(), content_type.lower()
    return {
        ("tiktok", "vídeo"): "21:00",
        ("tiktok", "imagem"): "20:00",
        ("instagram", "imagem"): "18:00",
        ("instagram", "vídeo"): "19:00",
        ("youtube", "vídeo"): "17:00",
        ("youtube", "imagem"): "16:00"
    }.get((p, c), "15:00")

def generate_hashtags(text, platform):
    base_tags = ["#Globoplay", "#Novidades", "#Trend"]
    platform_tags = {
        "instagram": ["#InstagramReels", "#InstaTrend"],
        "tiktok": ["#TikTokTrend", "#ForYou"],
        "youtube": ["#YouTubeBrasil", "#VideoTrend"]
    }
    clean_desc = clean_text(text)
    keyword_tags = [f"#{word.capitalize()}" for word in keywords if word in clean_desc]
    return base_tags + platform_tags.get(platform.lower(), []) + keyword_tags[:3]

def suggest_tone_style(text):
    clean_desc = clean_text(text)
    tone = []
    if "humor" in clean_desc or "memes" in clean_desc: tone.append("Divertido e leve")
    if "música" in clean_desc: tone.append("Cultural e artístico")
    if "sustentabilidade" in clean_desc or "diversidade" in clean_desc: tone.append("Engajado e social")
    if "novela" in clean_desc or "reality" in clean_desc: tone.append("Storytelling emocional")
    if not tone: tone.append("Informativo e direto")
    return tone

def generate_recommendations(text, platform, content_type):
    clean_desc = clean_text(text)
    tfidf_vector = vectorizer.transform([clean_desc]).toarray()
    genz_match_score = sum([1 for word in keywords if word in clean_desc])
    genz_affinity = platform_scores.get(platform.lower(), 0)
    text_length = len(text)
    num_hashtags = text.count("#")
    hour_score = int(get_best_time(platform, content_type).split(":")[0])
    num_features = np.array([[genz_match_score, genz_affinity, text_length, num_hashtags, hour_score]])
    num_features_scaled = scaler.transform(num_features)
    expected_tfidf_size = model.n_features_in_ - num_features_scaled.shape[1]
    tfidf_vector = tfidf_vector[:, :expected_tfidf_size]
    if tfidf_vector.shape[1] < expected_tfidf_size:
        tfidf_vector = np.pad(tfidf_vector, ((0, 0), (0, expected_tfidf_size - tfidf_vector.shape[1])), mode='constant')
    X_input = np.concatenate([num_features_scaled, tfidf_vector], axis=1)
    predicted_engagement = model.predict(X_input)[0]
    cta_options = ["Descubra mais no Globoplay!", "Confira tudo no Globoplay!", "Não perca essa novidade no Globoplay!"]
    cta = np.random.choice(cta_options)
    base_text = text.strip()
    optimized_text = base_text if any(phrase.lower() in base_text.lower() for phrase in [cta.lower(), "descubra mais no globoplay", "confira tudo no globoplay"]) else f"{base_text} | {cta}"
    tone_variations = {"tiktok": "Divertido e dinâmico", "instagram": "Visual e envolvente", "youtube": "Informativo e detalhado", "twitter": "Curto e impactante"}
    platform_tone = tone_variations.get(platform.lower(), "Informativo e direto")
    hashtags = generate_hashtags(text, platform)
    return {
        "platform": platform.capitalize(),
        "format": content_type.capitalize(),
        "hashtags": hashtags,
        "optimized_text": optimized_text.strip(),
        "best_time": get_best_time(platform, content_type),
        "tone_style": [platform_tone] + suggest_tone_style(text),
        "predicted_engagement": round(float(np.expm1(predicted_engagement)), 2)
    }

def generate_variation(text, platform, content_type):
    result_a = generate_recommendations(text, platform, content_type)
    alt_platform = "TikTok" if platform.lower() != "tiktok" else "Instagram"
    alt_format = "vídeo" if content_type.lower() != "vídeo" else "imagem"
    alt_text = f"{text.strip()} | {np.random.choice(['Confira tudo no Globoplay!', 'Não perca essa novidade no Globoplay!', 'Descubra mais no Globoplay!'])} {np.random.choice(['Você sabia que isso está chegando?', 'Prepare-se para algo novo!', 'Essa novidade vai te surpreender!'])}"
    result_b = generate_recommendations(alt_text, alt_platform, alt_format)
    return result_a, result_b

def generate_explanation(context, data):
    prompt = f"""
    Você é um assistente estratégico para campanhas digitais da Globo, focado em engajamento da Geração Z.
    Sua tarefa: {context}.
    Explique em até 150 palavras, com linguagem clara, tom positivo e 1 emoji no final.
    Evite termos técnicos.
    Dados da sugestão: {data}
    """
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"}
        )
        response_data = response.json()
        explanation = response_data.get("choices", [{}])[0].get("message", {}).get("content") \
                      or response_data.get("choices", [{}])[0].get("text")
        return explanation or "Explicação não disponível."
    except Exception as e:
        return f"Erro ao gerar explicação: {str(e)}"

# Endpoints
@app.post("/chat_interaction")
def chat_interaction(data: ChatInput):
    db = SessionLocal()
    suggestion_id = f"suggestion_{uuid4()}"
    result_a, result_b = generate_variation(data.text, data.platform, data.content_type)
    explanation_a = generate_explanation("Sugestão A", result_a)
    explanation_b = generate_explanation("Sugestão B", result_b)
    record = History(id=suggestion_id, created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id=data.user_id, input=json.dumps(data.dict()), output=json.dumps({"A": result_a, "B": result_b, "explanation_A": explanation_a, "explanation_B": explanation_b}))
    db.add(record)
    db.commit()
    db.close()
    return {"status": "success", "suggestion_id": suggestion_id, "suggestions": {"A": result_a, "B": result_b}, "explanations": {"A": explanation_a, "B": explanation_b}}

@app.post("/chat_response")
def chat_response(data: ChatResponse):
    db = SessionLocal()
    record = db.query(History).filter(
        History.id == data.suggestion_id,
        History.user_id == data.user_id  # Verifica se pertence ao usuário
    ).first()
    if record:
        record.chosen_version = data.chosen_version
        db.commit()
        db.close()
        return {"status": "success", "message": "Escolha registrada."}
    db.close()
    return {"status": "error", "message": "Registro não encontrado ou usuário inválido."}

@app.post("/chat_feedback")
def chat_feedback(data: ChatFeedback):
    db = SessionLocal()
    record = db.query(History).filter(
        History.id == data.suggestion_id,
        History.user_id == data.user_id  # Verifica se pertence ao usuário
    ).first()
    if record:
        record.feedback_reason = data.feedback
        db.commit()
        db.close()
        return {"status": "success", "message": "Feedback registrado."}
    db.close()
    return {"status": "error", "message": "Registro não encontrado ou usuário inválido."}

@app.get("/chat_export/{suggestion_id}")
def chat_export(suggestion_id: str):
    db = SessionLocal()
    record = db.query(History).filter(History.id == suggestion_id).first()
    db.close()
    if not record:
        return {"error": "ID não encontrado."}
    pdf = fitz.open()
    page = pdf.new_page()
    y = 72
    page.insert_text((72, y), f"Relatório - {suggestion_id}", fontsize=16); y += 30
    input_data = json.loads(record.input)
    output_data = json.loads(record.output)
    page.insert_text((72, y), f"Ideia original: {input_data['text']}"); y += 20
    page.insert_text((72, y), f"Plataforma: {input_data['platform']} | Formato: {input_data['content_type']}"); y += 20
    for version in ["A", "B"]:
        page.insert_text((72, y), f"--- Sugestão {version} ---", fontsize=12); y += 20
        for k, v in output_data[version].items():
            page.insert_text((90, y), f"{k}: {v}", fontsize=10); y += 14
        page.insert_text((90, y), f"Explicação: {output_data[f'explanation_{version}']}", fontsize=10); y += 20
    if record.chosen_version:
        page.insert_text((72, y), f"Escolha do usuário: {record.chosen_version}", fontsize=12); y += 20
    if record.feedback_reason:
        page.insert_text((72, y), f"Feedback: {record.feedback_reason}", fontsize=12); y += 20
    pdf_path = f"{suggestion_id}_relatorio.pdf"
    pdf.save(pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path)

@app.get("/history/{user_id}")
def get_user_history(user_id: str):
    db = SessionLocal()
    records = db.query(History).filter(History.user_id == user_id).all()
    db.close()
    return [
        {
            "id": r.id,
            "created_at": r.created_at,
            "input": json.loads(r.input),
            "output": json.loads(r.output),
            "chosen_version": r.chosen_version,
            "feedback_reason": r.feedback_reason
        }
        for r in records
    ]