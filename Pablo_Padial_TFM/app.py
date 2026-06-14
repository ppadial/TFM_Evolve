import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import datetime as dt

# === CONFIGURACIÓN DE LA PÁGINA ===
st.set_page_config(
    page_title="BiciMad · Predicción de Ocupación",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === ESTILOS PERSONALIZADOS ===
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stHeader"] {height: 0; visibility: hidden;}

    .block-container {
        padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1500px;
    }
    [data-testid="stSidebar"] {
        background-color: #FAFAFA; border-right: 1px solid #E5E5EA;
    }
    h1 { font-weight: 700; letter-spacing: -0.03em; color: #1D1D1F; }
    h2, h3 { color: #1D1D1F; font-weight: 600; }

    [data-testid="stMetric"] {
        background-color: #FAFAFA; border: 1px solid #E5E5EA;
        border-radius: 12px; padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] p { color: #6E6E73; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { font-weight: 700; color: #1D1D1F; }

    [data-testid="stAlert"] { border-radius: 12px; border: none; }

    .stButton > button {
        border-radius: 8px; font-weight: 500; border: 1px solid #E5E5EA;
        transition: all 0.15s ease;
    }
    .stButton > button:hover { border-color: #E30613; color: #E30613; }
    [data-baseweb="select"] > div { border-radius: 8px; }

    .id-badge {
        display: inline-block; background-color: #E30613; color: white;
        padding: 3px 10px; border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.8rem; font-weight: 600;
    }
    .subtitle {
        color: #6E6E73; font-size: 1.05rem;
        margin-top: -0.5rem; margin-bottom: 1.2rem;
    }
    .empty-state { text-align: center; padding: 32px 16px; color: #6E6E73; }
    .empty-state .emoji { font-size: 3.5rem; margin-bottom: 8px; }
    .empty-state .title { font-weight: 600; color: #1D1D1F; margin-bottom: 4px; }

    /* Filas de los rankings Top 5 */
    .rank-row {
        display: flex; align-items: center; gap: 8px;
        padding: 6px 8px; border-radius: 8px; margin-bottom: 4px;
        background-color: #FAFAFA; border: 1px solid #EFEFF1;
    }
    .rank-pos {
        font-weight: 700; font-size: 0.8rem; color: #6E6E73;
        min-width: 18px;
    }
    .rank-name {
        flex: 1; font-size: 0.85rem; color: #1D1D1F;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .rank-val {
        font-family: ui-monospace, monospace; font-size: 0.8rem;
        font-weight: 600;
    }
    .legend {
        font-size: 0.78rem; color: #6E6E73; line-height: 1.6;
    }
    .legend .dot {
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 5px; vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

PLACEHOLDER = "— Ninguna —"

# Etiquetas de las categorías que predice el modelo
CAT_BAJA, CAT_NORMAL, CAT_ALTA = 0, 1, 2
CAT_NOMBRE = {CAT_BAJA: "VACÍA (riesgo)", CAT_NORMAL: "Equilibrada", CAT_ALTA: "LLENA (riesgo)"}
CAT_COLOR  = {CAT_BAJA: "#E30613", CAT_NORMAL: "#27AE60", CAT_ALTA: "#F39C12"}


# === CARGA DE DATOS ===
@st.cache_data
def load_predicciones() -> pd.DataFrame:
    """
    Predicciones del modelo LightGBM (Optuna) sobre el periodo de validación.

    Cada fila es la predicción para una estación en una fecha-hora concreta:
      - pred_categoria: 0=BAJA (vacía), 1=NORMAL, 2=ALTA (llena)
      - prob_baja / prob_normal / prob_alta: confianza del modelo
      - ocupation: ocupación REAL de ese momento (para comparar en la demo)
    """
    df = pd.read_parquet("predicciones_app.parquet")
    df["hora"] = pd.to_datetime(df["hora"])
    df["fecha"] = df["hora"].dt.date
    return df


@st.cache_data
def load_stations(preds: pd.DataFrame) -> pd.DataFrame:
    """Catálogo único de estaciones (derivado de las predicciones)."""
    return (
        preds.drop_duplicates(subset=["id"])
             .dropna(subset=["latitude", "longitude"])
             .sort_values("id").reset_index(drop=True)
             [["id", "name", "latitude", "longitude", "total_bases"]]
             .rename(columns={"latitude": "lat", "longitude": "lon"})
    )


preds = load_predicciones()
stations = load_stations(preds)

# Rango de fechas disponible en las predicciones
FECHA_MIN = preds["fecha"].min()
FECHA_MAX = preds["fecha"].max()


def pred_for_datetime(fecha: dt.date, hour: int) -> pd.DataFrame:
    """Predicción de cada estación para una fecha y hora concretas."""
    sub = preds[(preds["fecha"] == fecha) & (preds["hora_dia"] == hour)]
    cols = ["id", "pred_categoria", "prob_baja", "prob_normal", "prob_alta", "ocupation"]
    return stations.merge(sub[cols], on="id", how="left")


def cat_color(cat) -> str:
    """Color según la categoría predicha."""
    if pd.isna(cat):
        return "#9E9E9E"
    return CAT_COLOR.get(int(cat), "#9E9E9E")


# === ESTADO ===
st.session_state.setdefault("selected_id", None)
st.session_state.setdefault("last_processed_click", None)


def on_search_change():
    label = st.session_state.station_search
    st.session_state.selected_id = (
        None if label == PLACEHOLDER else int(label.split(" - ", 1)[0])
    )


def select_station(station_id: int):
    st.session_state.selected_id = station_id
    match = stations[stations.id == station_id]
    if not match.empty:
        st.session_state.station_search = (
            f"{int(match.iloc[0].id):04d} - {match.iloc[0]['name']}"
        )


# === SIDEBAR: FILTROS ===
with st.sidebar:
    st.markdown("### 🚲 BiciMad")
    st.caption("Predicción de ocupación · Madrid")
    st.divider()

    st.markdown("**📅 Fecha objetivo**")
    fecha = st.date_input(
        "fecha", value=FECHA_MIN, min_value=FECHA_MIN, max_value=FECHA_MAX,
        label_visibility="collapsed", format="DD/MM/YYYY",
    )

    st.markdown("**⏰ Hora**")
    hour = st.slider("hour", 0, 23, 8, step=1, format="%d:00",
                     label_visibility="collapsed")

    st.markdown("**🔍 Estación**")
    option_labels = [PLACEHOLDER] + [
        f"{int(r.id):04d} - {r.name}" for r in stations.itertuples()
    ]
    if "station_search" not in st.session_state:
        st.session_state.station_search = PLACEHOLDER
    # Aplicar (antes de crear el widget) cualquier cambio pendiente que venga
    # de un clic en el mapa en la ejecución anterior.
    if "pending_search" in st.session_state:
        st.session_state.station_search = st.session_state.pop("pending_search")
    st.selectbox("search", option_labels,
                 key="station_search", on_change=on_search_change,
                 label_visibility="collapsed",
                 help="Escribe el ID o parte del nombre")

    if st.session_state.selected_id is not None:
        if st.button("✕ Limpiar selección", use_container_width=True):
            st.session_state.selected_id = None
            st.session_state.last_processed_click = None
            st.session_state.pending_search = PLACEHOLDER
            st.rerun()

    st.divider()

    # --- Rankings Top 5 a la fecha-hora seleccionada ---
    pred_now = pred_for_datetime(fecha, hour).dropna(subset=["pred_categoria"])

    st.markdown("**🔴 Mayor riesgo de quedarse VACÍA**")
    # Ordenamos por probabilidad de estar vacía (BAJA)
    top_empty = pred_now.nlargest(5, "prob_baja")
    for i, r in enumerate(top_empty.itertuples(), 1):
        st.button(
            f"{i}.  {r.name}  ·  riesgo {r.prob_baja*100:.0f}%",
            key=f"empty_{r.id}",
            on_click=select_station, args=(int(r.id),),
            use_container_width=True,
        )

    st.markdown("**🟠 Mayor riesgo de quedarse LLENA**")
    top_full = pred_now.nlargest(5, "prob_alta")
    for i, r in enumerate(top_full.itertuples(), 1):
        st.button(
            f"{i}.  {r.name}  ·  riesgo {r.prob_alta*100:.0f}%",
            key=f"full_{r.id}",
            on_click=select_station, args=(int(r.id),),
            use_container_width=True,
        )

    st.divider()
    st.caption(f"📍 {len(stations)} estaciones · datos {FECHA_MIN:%d/%m/%Y}–{FECHA_MAX:%d/%m/%Y}")


# === CABECERA ===
st.title("Predicción de Ocupación")
if st.session_state.selected_id is not None:
    sh = stations[stations.id == st.session_state.selected_id].iloc[0]
    st.markdown(f"<p class='subtitle'><strong>{sh['name']}</strong> · "
                f"predicción para el {fecha:%d/%m/%Y} a las {hour:02d}:00</p>",
                unsafe_allow_html=True)
else:
    st.markdown(f"<p class='subtitle'>Vista general · {fecha:%d/%m/%Y} "
                f"a las {hour:02d}:00</p>", unsafe_allow_html=True)


# === LAYOUT: MAPA (centro) + RESULTADOS (derecha) ===
col_map, col_info = st.columns([2, 1], gap="medium")

pred_map = pred_for_datetime(fecha, hour)

with col_map:
    if st.session_state.selected_id is not None:
        sel = stations[stations.id == st.session_state.selected_id].iloc[0]
        center, zoom = [sel.lat, sel.lon], 15
    else:
        center, zoom = [40.4220, -3.7038], 12

    m = folium.Map(location=center, zoom_start=zoom,
                   tiles="cartodbpositron", control_scale=True)

    for r in pred_map.itertuples():
        is_sel = (r.id == st.session_state.selected_id)
        base_color = cat_color(r.pred_categoria)
        if pd.isna(r.pred_categoria):
            etiqueta = "s/d"
        else:
            etiqueta = CAT_NOMBRE[int(r.pred_categoria)]
        folium.CircleMarker(
            location=[r.lat, r.lon],
            radius=12 if is_sel else 6,
            tooltip=r.name,
            popup=folium.Popup(
                f"<div style='font-family:sans-serif;font-size:13px;'>"
                f"<b>{r.name}</b><br>"
                f"<span style='color:#6E6E73;'>ID {int(r.id):04d} · "
                f"{etiqueta}</span></div>", max_width=240),
            color="#1D1D1F" if is_sel else base_color,
            fill=True, fill_color=base_color,
            fill_opacity=0.95 if is_sel else 0.7,
            weight=3 if is_sel else 1,
        ).add_to(m)

    map_data = st_folium(m, width=None, height=600,
                         returned_objects=["last_object_clicked_tooltip"],
                         key="map")

    # Leyenda de colores
    st.markdown(
        "<div class='legend'>"
        "<span class='dot' style='background:#E30613;'></span>Riesgo vacía < 25%&nbsp;&nbsp;"
        "<span class='dot' style='background:#27AE60;'></span>Equilibrada &nbsp;&nbsp;"
        "<span class='dot' style='background:#F39C12;'></span>Riesgo llena > 75%&nbsp;&nbsp;"
        "<span class='dot' style='background:#9E9E9E;'></span>Sin dato"
        "</div>", unsafe_allow_html=True)

    clicked = map_data.get("last_object_clicked_tooltip") if map_data else None
    if clicked and clicked != st.session_state.last_processed_click:
        st.session_state.last_processed_click = clicked
        match = stations[stations["name"] == clicked]
        if not match.empty:
            new_id = int(match.iloc[0].id)
            if new_id != st.session_state.selected_id:
                st.session_state.selected_id = new_id
                # No modificamos station_search aquí (el widget ya existe en esta
                # ejecución). Guardamos el valor pendiente y lo aplicamos al inicio
                # de la próxima ejecución, antes de crear el selectbox.
                st.session_state.pending_search = (
                    f"{new_id:04d} - {match.iloc[0]['name']}"
                )
                st.rerun()


with col_info:
    st.markdown("#### 📊 Resultados")
    if st.session_state.selected_id is not None:
        s = stations[stations.id == st.session_state.selected_id].iloc[0]
        pred_row = pred_map[pred_map.id == s.id].iloc[0]
        cat = pred_row.pred_categoria

        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;margin-bottom:4px;'>"
                f"<h3 style='margin:0;'>{s['name']}</h3>"
                f"<span class='id-badge'>#{int(s.id):04d}</span></div>",
                unsafe_allow_html=True)
            st.caption(f"📍 {s.lat:.5f}, {s.lon:.5f}")
            st.write("")

            c1, c2 = st.columns(2)
            c1.metric("Fecha", f"{fecha:%d/%m}")
            c2.metric("Hora", f"{hour:02d}:00")

            st.write("")
            st.markdown("**🔮 Predicción del modelo**")
            if pd.isna(cat):
                st.info("Sin predicción para esta franja.")
            else:
                cat = int(cat)
                probs = {
                    CAT_BAJA: pred_row.prob_baja,
                    CAT_NORMAL: pred_row.prob_normal,
                    CAT_ALTA: pred_row.prob_alta,
                }
                conf = probs[cat] * 100

                # Alerta según categoría predicha
                if cat == CAT_BAJA:
                    st.error(f"⚠️ Riesgo de estación **VACÍA** — priorizar reposición")
                elif cat == CAT_ALTA:
                    st.warning(f"⚠️ Riesgo de estación **LLENA** — priorizar retirada")
                else:
                    st.success("✅ Ocupación equilibrada prevista")

                st.caption(f"Confianza del modelo: {conf:.0f}%")

                # Barras de probabilidad de cada clase
                st.write("")
                st.markdown("**Probabilidades**")
                pb1, pb2, pb3 = st.columns(3)
                pb1.metric("Vacía", f"{pred_row.prob_baja*100:.0f}%")
                pb2.metric("Equilibrada", f"{pred_row.prob_normal*100:.0f}%")
                pb3.metric("Llena", f"{pred_row.prob_alta*100:.0f}%")

            st.caption("Predicción del modelo LightGBM optimizado.")
    else:
        with st.container(border=True):
            st.markdown(
                "<div class='empty-state'>"
                "<div class='emoji'>🗺️</div>"
                "<div class='title'>Sin estación seleccionada</div>"
                "<div style='font-size:0.9rem;'>Haz clic en el mapa<br>"
                "o usa el buscador de la barra lateral</div></div>",
                unsafe_allow_html=True)
