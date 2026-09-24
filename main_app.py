"""Laboratorio de LLMs con Groq y analisis local de texto."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tiktoken import get_encoding


st.set_page_config(page_title="Laboratorio LLM", page_icon="🧪", layout="wide")

st.markdown(
	"""
	<style>
	.block-container { max-width: 1450px; padding-top: 2rem; }
	[data-testid="stMetricValue"] { color: #0b6e69; }
	.hero { border-left: 5px solid #f4a261; padding: .2rem 1rem; margin-bottom: 1.5rem; }
	.hero h1 { margin-bottom: .25rem; }
	</style>
	""",
	unsafe_allow_html=True,
)

MODEL_OPTIONS = {
	"Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
	"Llama 3.1 8B Instant": "llama-3.1-8b-instant",
	"Qwen Qwen3 32B": "qwen/qwen3-32b",
	"Gemma2 9B IT": "gemma2-9b-it",
	"GPT OSS 120B": "openai/gpt-oss-120b",
	"GPT OSS 20B": "openai/gpt-oss-20b",
}


def tokenize(text: str) -> list[int]:
	return get_encoding("cl100k_base").encode(text)


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
	if not np.any(left) or not np.any(right):
		return 0.0
	return float(cosine_similarity(left.reshape(1, -1), right.reshape(1, -1))[0, 0])


def word_metrics(text_a: str, text_b: str) -> dict[str, float]:
	words_a = set(text_a.lower().split())
	words_b = set(text_b.lower().split())
	union = words_a | words_b
	intersection = words_a & words_b
	jaccard = len(intersection) / len(union) if union else 0.0
	vectorizer = TfidfVectorizer().fit([text_a, text_b])
	vectors = vectorizer.transform([text_a, text_b]).toarray()
	return {"Coseno TF-IDF": safe_cosine(vectors[0], vectors[1]), "Jaccard": jaccard}


def display_token_analysis(text: str) -> None:
	token_ids = tokenize(text)
	encoding = get_encoding("cl100k_base")
	token_text = [encoding.decode([token_id]) for token_id in token_ids]
	first_col, second_col = st.columns([1, 2])
	with first_col:
		st.metric("Tokens", len(token_ids))
		st.metric("Caracteres", len(text))
		st.metric("Tokens / palabra", round(len(token_ids) / max(len(text.split()), 1), 2))
	with second_col:
		st.dataframe(
			{"Posicion": range(len(token_ids)), "Token ID": token_ids, "Fragmento": token_text},
			use_container_width=True,
			hide_index=True,
		)


def generation_controls() -> tuple[str, dict[str, Any]]:
	selected_label = st.selectbox("Modelo Groq", list(MODEL_OPTIONS))
	st.caption(f"ID del modelo: `{MODEL_OPTIONS[selected_label]}`")
	temperature = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.05)
	max_tokens = st.slider("Max tokens", 64, 4096, 512, 64)
	top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
	system_prompt = st.text_area(
		"Instruccion del sistema",
		"Eres un asistente preciso. Responde en espanol y explica tus supuestos.",
		height=100,
	)
	return MODEL_OPTIONS[selected_label], {
		"temperature": temperature,
		"max_tokens": max_tokens,
		"top_p": top_p,
		"system_prompt": system_prompt,
	}


def main() -> None:
	st.markdown(
		'<div class="hero"><h1>Laboratorio de LLMs</h1>'
		'<p>Explora generacion, tokenizacion, representaciones vectoriales y similitud textual.</p></div>',
		unsafe_allow_html=True,
	)
	with st.sidebar:
		st.header("Conexion")
		api_key = st.text_input("API key de GROQ", type="password", help="Se usa solo durante esta sesion.")
		if api_key:
			st.success("API key lista")
		else:
			st.info("Ingresa tu API key para habilitar la generacion.")
		st.divider()
		st.caption("Los analisis de texto funcionan localmente.")

	tabs = st.tabs(["Generacion", "Tokens", "Bolsa de palabras", "Embeddings y similitud"])
	with tabs[0]:
		st.subheader("Generacion de texto")
		left, right = st.columns([1, 1.6], gap="large")
		with left:
			model, params = generation_controls()
		with right:
			prompt = st.text_area(
				"Prompt del usuario",
				"Explica en cinco puntos como evaluar un modelo de lenguaje.",
				height=180,
			)
			generate = st.button("Generar respuesta", type="primary", use_container_width=True)
			if generate:
				if not api_key:
					st.error("Ingresa una API key de GROQ en la barra lateral.")
				elif not prompt.strip():
					st.warning("Escribe un prompt antes de generar.")
				else:
					try:
						client = Groq(api_key=api_key)
						response = client.chat.completions.create(
							model=model,
							messages=[
								{"role": "system", "content": params["system_prompt"]},
								{"role": "user", "content": prompt},
							],
							temperature=params["temperature"],
							max_tokens=params["max_tokens"],
							top_p=params["top_p"],
						)
						st.session_state["last_response"] = response.choices[0].message.content or ""
						usage = response.usage
						if usage:
							st.caption(
								f"Tokens de prompt: {usage.prompt_tokens} · "
								f"Tokens de salida: {usage.completion_tokens} · Total: {usage.total_tokens}"
							)
					except Exception as error:
						st.error(f"No fue posible generar la respuesta: {error}")
			if st.session_state.get("last_response"):
				st.markdown("#### Respuesta")
				st.write(st.session_state["last_response"])

	with tabs[1]:
		st.subheader("Tokens y Token IDs")
		token_text = st.text_area(
			"Texto para tokenizar",
			"Los modelos de lenguaje procesan texto como secuencias de tokens.",
			key="token_text",
			height=140,
		)
		if token_text:
			display_token_analysis(token_text)

	with tabs[2]:
		st.subheader("Bolsa de palabras")
		bow_text = st.text_area(
			"Documentos (uno por linea)",
			"Los modelos aprenden patrones del lenguaje.\nLa tokenizacion transforma el lenguaje.",
			height=140,
		)
		documents = [line.strip() for line in bow_text.splitlines() if line.strip()]
		if documents:
			vectorizer = CountVectorizer(ngram_range=(1, 2))
			matrix = vectorizer.fit_transform(documents)
			terms = vectorizer.get_feature_names_out()
			bow = matrix.toarray()
			st.metric("Vocabulario", len(terms))
			st.dataframe(
				bow,
				column_config={str(index): terms[index] for index in range(len(terms))},
				use_container_width=True,
				hide_index=True,
			)
			frequencies = bow.sum(axis=0)
			top_indices = np.argsort(frequencies)[::-1][:10]
			st.bar_chart({terms[index]: int(frequencies[index]) for index in top_indices})

	with tabs[3]:
		st.subheader("Embeddings y metricas de similitud")
		st.caption("Los vectores TF-IDF funcionan como embeddings interpretables y se calculan localmente.")
		text_a = st.text_area("Texto A", "Los embeddings representan el significado de un texto.", height=100)
		text_b = st.text_area("Texto B", "Un vector puede capturar relaciones entre palabras y documentos.", height=100)
		if text_a and text_b:
			vectorizer = TfidfVectorizer()
			embedding_matrix = vectorizer.fit_transform([text_a, text_b]).toarray()
			metrics = word_metrics(text_a, text_b)
			metric_cols = st.columns(len(metrics))
			for column, (name, value) in zip(metric_cols, metrics.items()):
				column.metric(name, f"{value:.3f}")
			st.markdown("#### Embeddings TF-IDF")
			st.dataframe(
				embedding_matrix,
				column_config={str(index): term for index, term in enumerate(vectorizer.get_feature_names_out())},
				use_container_width=True,
				hide_index=True,
			)
			norm_a = math.sqrt(float(np.dot(embedding_matrix[0], embedding_matrix[0])))
			st.caption(f"Dimensiones del embedding: {embedding_matrix.shape[1]} · Norma A: {norm_a:.3f}")


if __name__ == "__main__":
	main()
