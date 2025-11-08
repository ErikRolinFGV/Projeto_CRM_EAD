import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# =========================
# 1. Carregar os dados
# =========================
df_ml = pd.read_csv("data/MachineLearn_Completo.csv")
df_genz = pd.read_csv("data/Preferências da Geração Z (respostas) - Respostas ao formulário 1(1).csv")

# =========================
# 2. Limpeza básica do texto
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@[\w_]+", "", text)
    text = re.sub(r"#[\w_]+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df_ml['clean_description'] = df_ml['description'].apply(clean_text)

# =========================
# 3. TF-IDF limitado a 50 palavras
# =========================
vectorizer = TfidfVectorizer(max_features=50)
tfidf_matrix = vectorizer.fit_transform(df_ml['clean_description'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# =========================
# 4. Codificar plataforma e autor
# =========================
le_platform = LabelEncoder()
df_ml['platform_encoded'] = le_platform.fit_transform(df_ml['source_platform'].astype(str))

le_author = LabelEncoder()
df_ml['author_encoded'] = le_author.fit_transform(df_ml['author'].astype(str))

# =========================
# 5. Criar score de afinidade com Geração Z por plataforma
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

# Mapear score para cada linha
platform_score_list = []
for p in df_ml['source_platform'].astype(str).str.lower():
    score = platform_scores.get(p, 0)
    platform_score_list.append(score)
df_ml['genz_affinity'] = platform_score_list

# =========================
# 6. Dataset final
# =========================
final_df = pd.concat([df_ml[['engagement_score', 'platform_encoded', 'author_encoded', 'genz_affinity']], tfidf_df], axis=1)

# =========================
# 7. Separar treino e teste
# =========================
X = final_df.drop(columns=['engagement_score'])
y = final_df['engagement_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 8. Treinar modelo
# =========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# 9. Avaliação
# =========================
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# =========================
# 10. Importância das features
# =========================
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False).head(20)

# Gráfico de importância
fig = px.bar(importance_df, x='feature', y='importance', title='Importância das Features para Engajamento')
fig.write_image("data/importancia_features.png")

# =========================
# 11. Salvar dataset final e modelo
# =========================
final_df.to_csv("data/dataset_final_para_modelo.csv", index=False)

with open("models/modelo_engajamento.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/encoders.pkl", "wb") as f:
    pickle.dump({"platform": le_platform, "author": le_author, "vectorizer": vectorizer}, f)

# =========================
# 12. Relatório
# =========================
print(f"Modelo treinado com sucesso. R²: {r2:.4f}, RMSE: {rmse:.2f}")