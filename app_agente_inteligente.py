"""Agente inteligente educativo: noticias, inferencia bayesiana y criterio de Kelly.

Proyecto de Laboratorio de Sistemas Inteligentes (FIME-UANL).
Alineado con las practicas A2.1 (PEAS), A2.2 (arquitectura) y A3.1 (propuesta).
Ejecutar con: streamlit run app_agente_valor.py

ADVERTENCIA: La app no realiza apuestas ni garantiza
rendimientos; su finalidad es exclusivamente academica.
"""

from __future__ import annotations

import math
import requests
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------------------------------------------------------
# Constantes y ejemplos didacticos
# -----------------------------------------------------------------------------
RESULTADOS = ("Local", "Empate", "Visitante")
# Colores ajustados para armonizar con el diseño corporativo
COLORES = {"azul": "#3b82f6", "verde": "#10b981", "rojo": "#ef4444"}
EJEMPLOS = {
    "Favorables al local": (
        "El equipo local consiguió una victoria excelente. Su delantero está "
        "recuperado y disponible para jugar.",
        "El visitante sufrió una derrota terrible. Dos jugadores están "
        "lesionados y su capitán fue suspendido.",
    ),
    "Favorables al visitante": (
        "El local sufrió una derrota terrible. Su delantero está lesionado "
        "y descartado para el partido.",
        "El equipo visitante consiguió una victoria excelente y sus jugadores "
        "están recuperados y disponibles.",
    ),
    "Sin noticias": ("", ""),
}


@st.cache_data(show_spinner=False, ttl=120)
def obtener_cuotas_api(api_key: str, deporte: str) -> list | dict:
    """Consume The Odds API para obtener partidos en vivo de la liga indicada."""
    url = f"https://api.the-odds-api.com/v4/sports/{deporte}/odds"
    parametros = {
        "apiKey": api_key,
        "regions": "eu,us",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    try:
        respuesta = requests.get(url, params=parametros)
        return respuesta.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_resource(show_spinner=False)
def crear_analizador() -> SentimentIntensityAnalyzer:
    """Inicializa VADER una sola vez y amplía su léxico deportivo en español."""
    analizador = SentimentIntensityAnalyzer()
    analizador.lexicon.update(
        {
            "victoria": 2.8, "victorias": 2.8, "triunfo": 2.8,
            "ganó": 2.4, "ganaron": 2.4, "excelente": 3.0,
            "bueno": 1.9, "buena": 1.9, "mejor": 1.8,
            "recuperado": 2.3, "recuperada": 2.3,
            "recuperados": 2.3, "recuperadas": 2.3,
            "disponible": 1.5, "disponibles": 1.5,
            "derrota": -2.7, "derrotas": -2.7, "terrible": -3.0,
            "lesión": -2.8, "lesiones": -2.8,
            "lesionado": -2.8, "lesionada": -2.8,
            "lesionados": -2.8, "lesionadas": -2.8,
            "grave": -2.1, "graves": -2.1,
            "suspendido": -2.6, "suspendida": -2.6,
            "sancionado": -2.6, "sancionada": -2.6,
            "descartado": -2.4, "descartada": -2.4,
            "perdió": -2.3, "perdieron": -2.3,
            "baja": -1.3, "bajas": -1.3,
        }
    )
    return analizador


def analizar_noticia(texto: str, analizador: SentimentIntensityAnalyzer) -> float:
    """Devuelve polaridad compound de VADER, dentro de [-1.0, +1.0]."""
    if not texto.strip():
        return 0.0
    return float(analizador.polarity_scores(texto[:5000])["compound"])


def preparar_prior(
    cuotas: tuple[float, float, float],
    frecuencias: tuple[int, int, int],
    peso_historico: float,
) -> tuple[list[float], list[float], list[float], float]:
    """Obtiene probabilidades implícitas y un prior sin margen comercial."""
    if any((not math.isfinite(c)) or c <= 1.0 for c in cuotas):
        raise ValueError("Las tres cuotas decimales deben ser mayores que 1.0.")
    if any(f < 0 for f in frecuencias):
        raise ValueError("Las frecuencias históricas no pueden ser negativas.")
    if not 0.0 <= peso_historico <= 1.0:
        raise ValueError("El peso histórico debe estar entre 0 y 1.")

    implicitas = [1.0 / cuota for cuota in cuotas]
    suma_implicitas = sum(implicitas)
    mercado_justo = [p / suma_implicitas for p in implicitas]
    total_partidos = sum(frecuencias)

    if total_partidos == 0:
        historicas = mercado_justo.copy()
        peso_historico = 0.0
    else:
        historicas = [frecuencia / total_partidos for frecuencia in frecuencias]

    prior = [
        (1.0 - peso_historico) * p_mercado + peso_historico * p_historica
        for p_mercado, p_historica in zip(mercado_justo, historicas)
    ]
    return implicitas, mercado_justo, prior, suma_implicitas


def actualizar_bayes(
    prior: list[float],
    sentimiento_local: float,
    sentimiento_visitante: float,
    intensidad: float,
) -> tuple[list[float], list[float]]:
    """Aplica Bayes con razones de verosimilitud heurísticas por resultado."""
    if len(prior) != 3 or any(p < 0 for p in prior):
        raise ValueError("El prior debe contener tres probabilidades válidas.")
    if not math.isclose(sum(prior), 1.0, abs_tol=1e-8):
        raise ValueError("El prior debe sumar 1.")
    if not (-1 <= sentimiento_local <= 1 and -1 <= sentimiento_visitante <= 1):
        raise ValueError("Las polaridades deben pertenecer a [-1, 1].")
    if not 0 <= intensidad <= 1:
        raise ValueError("La intensidad debe pertenecer a [0, 1].")

    diferencia = sentimiento_local - sentimiento_visitante
    factores = [math.exp(intensidad * diferencia), 1.0,
                math.exp(-intensidad * diferencia)]
    numeradores = [p * factor for p, factor in zip(prior, factores)]
    normalizador = sum(numeradores)
    posterior = [valor / normalizador for valor in numeradores]
    return posterior, factores


def clasificar_riesgo(probabilidad: float) -> str:
    """Clasificación DIDÁCTICA de riesgo por probabilidad de perder."""
    probabilidad_perder = 1.0 - probabilidad
    if probabilidad_perder < 0.40: return "Bajo"
    if probabilidad_perder <= 0.65: return "Medio"
    return "Alto"


def evaluar_resultados(
    cuotas: tuple[float, float, float],
    implicitas: list[float],
    prior: list[float],
    posterior: list[float],
    fraccion_kelly: float,
    limite_stake: float,
) -> list[dict[str, Any]]:
    """Calcula EV y Kelly para los tres resultados mutuamente excluyentes."""
    filas: list[dict[str, Any]] = []
    for i, nombre in enumerate(RESULTADOS):
        cuota, p = cuotas[i], posterior[i]
        ev = p * cuota - 1.0
        kelly_completo = max(0.0, (cuota * p - 1.0) / (cuota - 1.0))
        stake = min(kelly_completo * fraccion_kelly, limite_stake)

        utilidad = (
            p * math.log1p(stake * (cuota - 1.0))
            + (1.0 - p) * math.log1p(-stake)
        ) if stake > 0.0 else 0.0

        filas.append({
            "resultado": nombre, "cuota": cuota, "implicita": implicitas[i],
            "prior": prior[i], "posterior": p, "ev": ev,
            "kelly_completo": kelly_completo, "stake": stake,
            "riesgo": clasificar_riesgo(p), "utilidad": utilidad, "es_valor": ev > 1e-10,
        })
    return filas


def elegir_recomendacion(filas: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Escoge UNA opción de valor con máxima utilidad logarítmica estimada."""
    candidatas = [f for f in filas if f["es_valor"] and f["stake"] > 0]
    return max(candidatas, key=lambda f: (f["utilidad"], f["ev"])) if candidatas else None


def porcentaje(numero: float, decimales: int = 1) -> str:
    """Muestra una proporción (0..1) en formato porcentual."""
    return f"{numero * 100:.{decimales}f}%"


def grafico_probabilidades(prior: list[float], posterior: list[float]) -> go.Figure:
    """Gráfica interactiva del antes y después de aplicar el módulo NLP."""
    fig = go.Figure()
    for etiqueta, valores, color in (
        ("Antes del NLP", prior, COLORES["azul"]),
        ("Después del NLP", posterior, COLORES["verde"]),
    ):
        fig.add_bar(
            name=etiqueta, x=list(RESULTADOS), y=valores,
            marker_color=color, text=[porcentaje(v) for v in valores],
            textposition="outside", hovertemplate="%{x}: %{y:.1%}<extra>" + etiqueta + "</extra>",
        )
    fig.update_layout(
        barmode="group", template="plotly_dark", height=370,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e5e7eb"),
        margin=dict(l=10, r=10, t=25, b=10),
        yaxis=dict(title="Probabilidad", tickformat=".0%", range=[0, 1.06], gridcolor="#262a35"),
        xaxis_title="Resultado", legend=dict(orientation="h", y=1.14),
    )
    return fig


def grafico_valor(filas: list[dict[str, Any]]) -> go.Figure:
    """Gráfica de valor esperado de cada decisión."""
    evs = [fila["ev"] for fila in filas]
    fig = go.Figure(
        go.Bar(
            x=list(RESULTADOS), y=[ev * 100 for ev in evs],
            marker_color=[COLORES["verde"] if x > 0 else COLORES["rojo"] for x in evs],
            text=[f"{x * 100:+.1f}%" for x in evs], textposition="outside",
            hovertemplate="%{x}: %{y:+.2f}% de EV<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#4b5563")
    fig.update_layout(
        template="plotly_dark", height=370, 
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e5e7eb"),
        margin=dict(l=10, r=10, t=25, b=10),
        yaxis=dict(title="Valor esperado por unidad apostada (%)", gridcolor="#262a35"),
        xaxis_title="Resultado", showlegend=False,
    )
    return fig


def tabla_resultados(filas: list[dict[str, Any]]) -> pd.DataFrame:
    """Convierte los cálculos en tabla clara y exportable."""
    return pd.DataFrame([
        {
            "Resultado": f["resultado"], "Cuota": f'{f["cuota"]:.2f}',
            "P. implícita": porcentaje(f["implicita"]),
            "P. previa": porcentaje(f["prior"]),
            "P. ajustada": porcentaje(f["posterior"]),
            "Valor esperado": f'{f["ev"] * 100:+.1f}%',
            "Kelly teórico": porcentaje(f["kelly_completo"]),
            "Stake prudente": porcentaje(f["stake"]), "Riesgo": f["riesgo"],
        } for f in filas
    ])


def main() -> None:
    """Construye el dashboard Streamlit; sus controles recalculan al cambiar."""
    st.set_page_config(
        page_title="Agente de IA | Value Bets (Laboratorio)",
        page_icon="🧠", layout="wide", initial_sidebar_state="expanded",
    )
    
    # ==============================================================================
    # INYECCIÓN DE CSS - DISEÑO CORPORATIVO FINTECH / INSTITUCIONAL (FIME UANL)
    # Mejorado para visibilidad completa y diseño elegante
    # ==============================================================================
    st.markdown("""
    <style>
        /* Importar fuente moderna Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Fondo general de la aplicación y tipografía global */
        .stApp {
            background-color: #121418 !important;
            font-family: 'Inter', sans-serif !important;
            color: #e5e7eb !important;
        }

        /* Ocultar o darle color oscuro a la barra superior (Header de Streamlit) */
        [data-testid="stHeader"] {
            background-color: #121418 !important;
        }

        /* Barra lateral (Sidebar) */
        [data-testid="stSidebar"] {
            background-color: #181b22 !important;
            border-right: 1px solid #262a35 !important;
        }

        /* Forzar visibilidad de textos, etiquetas y markdowns */
        h1, h2, h3, h4, h5, h6, span, p, label, .stMarkdown {
            color: #f3f4f6 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Ajuste fino para captions/textos secundarios (más elegantes) */
        .st-emotion-cache-16idsys p, .st-emotion-cache-1kyxreq p {
            color: #9ca3af !important; 
        }

        /* --- INPUTS: Text, Number, Selectbox --- */
        .stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div {
            background-color: #1a1d26 !important;
            color: #f3f4f6 !important;
            border: 1px solid #2a2e3d !important;
            border-radius: 6px !important;
            font-size: 14px !important;
        }
        input::placeholder, textarea::placeholder {
            color: #6b7280 !important;
        }
        .stTextInput input:focus, .stNumberInput input:focus, div[data-baseweb="select"] > div:focus-within {
            border-color: #3b82f6 !important;
            box-shadow: none !important;
        }

        /* --- TEXTAREAS (Noticias y NLP) --- */
        textarea {
            background-color: #1a1d26 !important;
            color: #f3f4f6 !important;
            border: 1px solid #2a2e3d !important;
            border-radius: 6px !important;
            font-size: 14px !important;
            padding: 10px !important;
        }
        textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: none !important;
        }

        /* --- TARJETAS DE MÉTRICAS --- */
        div[data-testid="metric-container"] {
            background-color: #181b22 !important;
            border: 1px solid #262a35 !important;
            border-radius: 8px !important;
            padding: 16px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 26px !important;
            font-weight: 700 !important;
            color: #ffffff !important;
        }
        [data-testid="stMetricLabel"] {
            color: #9ca3af !important;
            font-weight: 500 !important;
        }

        /* --- BOTONES PROFESIONALES --- */
        .stButton>button {
            background-color: #2563eb !important; 
            color: #ffffff !important;
            font-weight: 500 !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 6px 16px !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover {
            background-color: #1d4ed8 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        }
        .stButton>button:active {
            transform: translateY(0px) !important;
        }

        /* --- PESTAÑAS (Tabs) --- */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            color: #6b7280 !important;
            font-weight: 500 !important;
        }
        button[aria-selected="true"] {
            background-color: transparent !important;
            color: #f3f4f6 !important;
            border-bottom: 2px solid #3b82f6 !important;
        }

        /* --- ALERTAS --- */
        div[data-testid="stAlert"] {
            background-color: #064e3b !important;
            color: #a7f3d0 !important;
            border: 1px solid #047857 !important;
            border-radius: 6px !important;
        }
        
        /* Líneas divisorias (st.divider) */
        hr {
            border-bottom-color: #262a35 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Agente inteligente de métricas deportivas")
    st.caption("FIME · Laboratorio de Sistemas Inteligentes · Conexión a API en Vivo")

    # -------------------------------------------------------------------------
    # The Odds API - Integración en el sidebar
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("📡 The Odds API (Datos en vivo)")
        api_key = st.text_input("API Key", "28c3efeeae18594e1abaafdff1a4cf01", type="password")
        
        ligas_disponibles = {
            "Liga MX": "soccer_mexico_ligamx", 
            "Premier League": "soccer_epl", 
            "La Liga (España)": "soccer_spain_la_liga", 
            "UEFA Champions League": "soccer_uefa_champs_league"
        }
        liga_seleccionada = st.selectbox("Liga a consultar", list(ligas_disponibles.keys()))

        if st.button("Actualizar partidos en vivo", use_container_width=True):
            datos = obtener_cuotas_api(api_key, ligas_disponibles[liga_seleccionada])
            
            if isinstance(datos, dict) and "message" in datos:
                st.error(f"Error de API: {datos['message']}")
            elif isinstance(datos, dict) and "error" in datos:
                st.error(f"Error de red: {datos['error']}")
            elif isinstance(datos, list):
                partidos_procesados = {}
                for evento in datos:
                    local = evento.get("home_team", "")
                    visitante = evento.get("away_team", "")
                    bms = evento.get("bookmakers", [])
                    if bms:
                        bm = bms[0] # Tomar la primera casa de apuestas
                        mercados = bm.get("markets", [])
                        if mercados:
                            outcomes = mercados[0].get("outcomes", [])
                            c_l, c_e, c_v = 0.0, 0.0, 0.0
                            for o in outcomes:
                                if o["name"] == local: c_l = o["price"]
                                elif o["name"] == visitante: c_v = o["price"]
                                elif o["name"].lower() in ["draw", "empate"]: c_e = o["price"]

                            if c_l and c_e and c_v:
                                label = f"{local} vs {visitante} ({bm['title']})"
                                partidos_procesados[label] = {
                                    "l": local, "v": visitante, "cl": c_l, "ce": c_e, "cv": c_v
                                }
                st.session_state["partidos_api"] = partidos_procesados
                st.success(f"{len(partidos_procesados)} partidos cargados con éxito.")

        # Establecer variables por defecto (se sobreescriben si se selecciona un partido en vivo)
        def_l, def_v = "Equipo A", "Equipo B"
        def_cl, def_ce, def_cv = 2.30, 3.40, 3.10

        if st.session_state.get("partidos_api"):
            opciones = list(st.session_state["partidos_api"].keys())
            partido_elegido = st.selectbox("Selecciona un partido cargado", opciones)
            if partido_elegido:
                p_data = st.session_state["partidos_api"][partido_elegido]
                def_l, def_v = p_data["l"], p_data["v"]
                def_cl, def_ce, def_cv = p_data["cl"], p_data["ce"], p_data["cv"]
        
        st.divider()

        st.header("⚙️ Configuración del entorno")
        nombre_local = st.text_input("Nombre del equipo local", def_l).strip() or "Equipo A"
        nombre_visitante = st.text_input("Nombre del visitante", def_v).strip() or "Equipo B"

        st.subheader("Cuotas decimales (mercado 1X2)")
        cuota_local = st.number_input("Cuota local", min_value=1.01, max_value=100.0, value=float(def_cl), step=0.05)
        cuota_empate = st.number_input("Cuota empate", min_value=1.01, max_value=100.0, value=float(def_ce), step=0.05)
        cuota_visitante = st.number_input("Cuota visitante", min_value=1.01, max_value=100.0, value=float(def_cv), step=0.05)

        st.subheader("Historial comparable (opcional)")
        victorias_local = st.number_input("Victorias locales", 0, 10000, 12)
        empates = st.number_input("Empates", 0, 10000, 5)
        victorias_visitante = st.number_input("Victorias visitantes", 0, 10000, 7)
        peso_hist = st.slider("Peso del historial en el prior", 0, 100, 25, 5) / 100

        st.subheader("Parámetros del agente")
        intensidad = st.slider("Intensidad del impacto de noticias", 0.0, 1.0, 0.40, 0.05)
        fraccion_kelly = st.slider("Fracción de Kelly aplicada", 0.05, 1.0, 0.25, 0.05)
        limite_stake = st.slider("Máximo de capital por pronóstico (%)", 1, 20, 5, 1) / 100
        capital = st.number_input("Capital de simulación (MXN)", min_value=1.0, max_value=10_000_000.0, value=1000.0, step=100.0)

    # Botones de casos de prueba
    if "noticia_local" not in st.session_state:
        st.session_state["noticia_local"] = EJEMPLOS["Favorables al local"][0]
    if "noticia_visitante" not in st.session_state:
        st.session_state["noticia_visitante"] = EJEMPLOS["Favorables al local"][1]
    if "historial_agente" not in st.session_state:
        st.session_state["historial_agente"] = []

    st.subheader("1 · Sensores de texto y análisis NLP")
    botones = st.columns(3)
    for boton, (nombre, textos) in zip(botones, EJEMPLOS.items()):
        with boton:
            if st.button(nombre, use_container_width=True):
                st.session_state["noticia_local"] = textos[0]
                st.session_state["noticia_visitante"] = textos[1]

    entradas = st.columns(2)
    with entradas[0]:
        texto_local = st.text_area(f"Noticias de {nombre_local} (local)", key="noticia_local", height=140, max_chars=5000)
    with entradas[1]:
        texto_visitante = st.text_area(f"Noticias de {nombre_visitante} (visitante)", key="noticia_visitante", height=140, max_chars=5000)

    analizador = crear_analizador()
    s_local = analizar_noticia(texto_local, analizador)
    s_visitante = analizar_noticia(texto_visitante, analizador)

    cuotas = (cuota_local, cuota_empate, cuota_visitante)
    frecuencias = (victorias_local, empates, victorias_visitante)
    implicitas, mercado, prior, suma_implicitas = preparar_prior(cuotas, frecuencias, peso_hist)
    posterior, factores = actualizar_bayes(prior, s_local, s_visitante, intensidad)

    filas = evaluar_resultados(cuotas, implicitas, prior, posterior, fraccion_kelly, limite_stake)
    seleccion = elegir_recomendacion(filas)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"NLP · {nombre_local}", f"{s_local:+.3f}")
    m2.metric(f"NLP · {nombre_visitante}", f"{s_visitante:+.3f}")
    m3.metric("Suma de probabilidades finales", porcentaje(sum(posterior)))
    m4.metric("Margen implícito del mercado", f"{(suma_implicitas - 1) * 100:+.2f}%")

    st.subheader("2 · Inferencia bayesiana y visualización")
    col_prob, col_ev = st.columns(2)
    with col_prob: st.plotly_chart(grafico_probabilidades(prior, posterior), use_container_width=True)
    with col_ev: st.plotly_chart(grafico_valor(filas), use_container_width=True)

    st.subheader("3 · Modelo de utilidad y recomendación")
    if seleccion is None:
        st.warning("**RECOMENDACIÓN: NO APOSTAR.** Ninguno presenta valor esperado positivo.")
        stake_sugerido, accion, riesgo_final = 0.0, "No apostar", "Sin exposición"
    else:
        accion = (nombre_local if seleccion["resultado"] == "Local" else
                  nombre_visitante if seleccion["resultado"] == "Visitante" else "Empate")
        stake_sugerido, riesgo_final = seleccion["stake"], seleccion["riesgo"]
        st.success(f"**RECOMENDACIÓN DE SIMULACIÓN: {accion.upper()}** | Stake: **{porcentaje(stake_sugerido, 2)}** | Riesgo: **{riesgo_final}**")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Decisión del agente", accion)
    r2.metric("Stake sugerido", porcentaje(stake_sugerido, 2))
    r3.metric("Exposición simulada (MXN)", f"${capital * stake_sugerido:,.2f}")
    r4.metric("Nivel de riesgo", riesgo_final)

    tabla = tabla_resultados(filas)
    st.dataframe(tabla, hide_index=True, use_container_width=True)

    st.subheader("4 · Estado interno / registro de decisiones")
    if st.button("📌 Registrar análisis actual", type="primary"):
        st.session_state["historial_agente"].insert(0, {
            "Fecha y hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Local": nombre_local, "Visitante": nombre_visitante,
            "Cuotas L/E/V": "/".join(f"{c:.2f}" for c in cuotas),
            "NLP L/V": f"{s_local:+.2f}/{s_visitante:+.2f}",
            "Prob. L/E/V": "/".join(porcentaje(p) for p in posterior),
            "Decisión": accion, "Stake": porcentaje(stake_sugerido, 2), "Riesgo": riesgo_final,
        })
        st.session_state["historial_agente"] = st.session_state["historial_agente"][:10]
    
    if st.session_state["historial_agente"]:
        st.dataframe(pd.DataFrame(st.session_state["historial_agente"]), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()