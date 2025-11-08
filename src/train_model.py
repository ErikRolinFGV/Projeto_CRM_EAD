import pandas as pd
import numpy as np
import re
import string
import pickle
import os
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# =========================
# 0. Limpar pasta models antes de salvar novos arquivos
# =========================
models_dir = "models"
if os.path.exists(models_dir):
    shutil.rmtree(models_dir)  # Remove a pasta inteira
os.makedirs(models_dir)  # Cria novamente a pasta limpa

print("✅ Pasta 'models' limpa e pronta para salvar novos arquivos.")

# =========================
# 1. Carregar os dados
# =========================
df_ml = pd.read_csv("data/MachineLearn_Completo.csv")
df_final = pd.read_csv("data/dataset_final_para_modelo.csv")
df_genz = pd.read_csv("data/Preferências da Geração Z (respostas) - Respostas ao formulário 1(1).csv")

# =========================
# 2. Limpeza do texto
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"#\\w+", "", text)
    text = re.sub(r"https?://\\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\\s+", " ", text).strip()
    return text

df_ml["clean_description"] = df_ml["description"].apply(clean_text)

# =========================
# 3. TF-IDF expandido (300 termos)
# =========================
vectorizer = TfidfVectorizer(max_features=300)
tfidf_matrix = vectorizer.fit_transform(df_ml["clean_description"])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# =========================
# 4. Score Geração Z + Globo (por palavras-chave)
# =========================
keywords = [
    "humor", "música", "moda", "memes", "sustentabilidade", "diversidade", "games", "tecnologia", "lifestyle",
    "novela", "jornalismo", "esporte", "reality", "entretenimento", "cultura", "celebridade", "globoplay", "tvglobo"
]

df_ml["genz_match_score"] = df_ml["clean_description"].apply(
    lambda x: sum([1 for word in keywords if word in x])
)

# =========================
# 5. Afinidade por plataforma (usando dataset de preferências)
# =========================
platforms = df_genz['Quais plataformas você mais usa para descobrir novos conteúdos ou marcas?'].dropna().str.lower().str.split(',')
platform_scores = {}
for plist in platforms:
    for p in plist:
        p = p.strip()
        platform_scores[p] = platform_scores.get(p, 0) + 1

# Normalizar os scores
total = sum(platform_scores.values())
for k in platform_scores:
    platform_scores[k] /= total

# Mapear score para cada linha do df_ml
df_ml["genz_affinity"] = df_ml["source_platform"].astype(str).str.lower().map(lambda p: platform_scores.get(p, 0))

# =========================
# 6. Preparar target com log-transform
# =========================
df_final["log_engagement"] = np.log1p(df_final["engagement_score"])

# =========================
# 7. Unificar dataset final
# =========================
extra_features = df_ml[["genz_match_score", "genz_affinity"]]
model_df = pd.concat([df_final, extra_features, tfidf_df], axis=1)

# =========================
# 8. Separar X e y
# =========================
X = model_df.drop(columns=["engagement_score", "log_engagement"])
y = model_df["log_engagement"]

# Normalizar numéricas
num_cols = ["platform_encoded", "author_encoded", "genz_affinity", "genz_match_score"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Garantir que X seja numérico e sem duplicatas
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.loc[:, ~X.columns.duplicated()]

# =========================
# 9. Treinar modelo XGBoost com cross-validation
# =========================
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")

# Treinar modelo final
model.fit(X, y)

# =========================
# 10. Salvar pipeline
# =========================
with open(os.path.join(models_dir, "modelo_xgb.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(models_dir, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

# =========================
# 11. Relatório
# =========================
print(f"RMSE médio (cross-validation): {-scores.mean():.4f}")
print(f"Desvio padrão do RMSE: {scores.std():.4f}")
print("✅ Modelo XGBoost treinado e pipeline salvo com sucesso (com integração do dataset de preferências).")