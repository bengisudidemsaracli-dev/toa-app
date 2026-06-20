import os
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =====================================================
# SAYFA AYARI
# =====================================================

st.set_page_config(
    page_title="TOA Operasyon Gereksinimi Prototipi",
    page_icon="🩺",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #dff4ff;
    }
    [data-testid="stHeader"] {
        background-color: rgba(223, 244, 255, 0.88);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# DOSYA ADLARI
# =====================================================

DATA_FILE = "toa_anonymous_var_columns.csv"
MAPPING_FILE = "toa_variable_mapping_LOCAL_ONLY.csv"


# =====================================================
# ANONİM DEĞİŞKEN EŞLEŞTİRMESİ
# =====================================================

TARGET_COL = "var28"  # OP. GEREKSİNİMİ

VAR_MAP = {
    "var11": "WBC ilk",
    "var12": "WBC kontrol",
    "var13": "delta WBC",
    "var14": "NLR ilk",
    "var15": "NLR kontrol",
    "var16": "delta NLR",
    "var17": "CRP ilk",
    "var18": "CRP kontrol",
    "var19": "delta CRP",
    "var20": "Apse boyutu ilk",
    "var21": "Apse boyutu kontrol",
    "var22": "delta apse",
    "var28": "Operasyon gereksinimi",
}

DELTA_FEATURES = ["var13", "var16", "var19", "var22"]


# =====================================================
# CACHE
# =====================================================

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_FILE):
        return None
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_mapping():
    if not os.path.exists(MAPPING_FILE):
        return None
    return pd.read_csv(MAPPING_FILE)


@st.cache_data
def prepare_model_df(df):
    data = df.copy()

    needed_cols = [
        "var11", "var12", "var14", "var15",
        "var17", "var18", "var20", "var21",
        TARGET_COL
    ]

    for col in needed_cols:
        if col not in data.columns:
            raise ValueError(f"Eksik kolon: {col}")

    for col in needed_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Delta yönü: ilk - kontrol
    data["var13"] = data["var11"] - data["var12"]  # delta WBC
    data["var16"] = data["var14"] - data["var15"]  # delta NLR
    data["var19"] = data["var17"] - data["var18"]  # delta CRP
    data["var22"] = data["var20"] - data["var21"]  # delta apse

    model_df = data[DELTA_FEATURES + [TARGET_COL]].copy()

    for col in DELTA_FEATURES:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        model_df[col] = model_df[col].fillna(model_df[col].median())

    model_df[TARGET_COL] = pd.to_numeric(model_df[TARGET_COL], errors="coerce")
    model_df = model_df.dropna(subset=[TARGET_COL])
    model_df[TARGET_COL] = model_df[TARGET_COL].astype(int)

    return model_df


def get_model(model_name):
    if model_name == "Naive Bayes":
        return GaussianNB()

    if model_name == "Lojistik Regresyon":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42
        )

    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

    if model_name == "SVM":
        return SVC(
            C=0.1,
            kernel="linear",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )

    return GaussianNB()


@st.cache_resource
def train_and_evaluate_model(model_df, model_name):
    X = model_df[DELTA_FEATURES]
    y = model_df[TARGET_COL]

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify
    )

    base_model = get_model(model_name)

    eval_model = clone(base_model)
    eval_model.fit(X_train, y_train)

    y_pred = eval_model.predict(X_test)

    if hasattr(eval_model, "predict_proba"):
        y_prob = eval_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision 0": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        "Recall 0": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        "F1 0": f1_score(y_test, y_pred, pos_label=0, zero_division=0),
        "Precision 1": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall 1": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1 1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
    }

    try:
        metrics["AUC"] = roc_auc_score(y_test, y_prob)
    except Exception:
        metrics["AUC"] = np.nan

    try:
        metrics["Brier Skoru"] = brier_score_loss(y_test, y_prob)
    except Exception:
        metrics["Brier Skoru"] = np.nan

    roc_df = None
    if y_test.nunique() == 2:
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_df = pd.DataFrame({
                "Yanlış pozitif oranı": fpr,
                "Doğru pozitif oranı": tpr,
                "Eşik": thresholds,
            })
        except Exception:
            roc_df = None

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Gerçek 0", "Gerçek 1"],
        columns=["Tahmin 0", "Tahmin 1"]
    )

    # Tahmin için final model tüm veriyle yeniden eğitilir
    final_model = clone(base_model)
    final_model.fit(X, y)

    return final_model, metrics, cm_df, roc_df


@st.cache_data
def get_risk_factor_options():
    return {
        "Klinik öykü": [
            "Geçirilmiş PID öyküsü",
            "Daha önce tubo-ovaryan apse öyküsü",
            "Geçirilmiş jinekolojik/pelvik cerrahi öyküsü",
            "Endometriozis / endometrioma öyküsü",
            "İnfertilite tedavisi / yardımcı üreme tekniği öyküsü",
        ],
        "Enfeksiyon ve davranışsal riskler": [
            "Cinsel yolla bulaşan enfeksiyon öyküsü",
            "Yeni veya çoklu cinsel partner öyküsü",
            "Korunmasız cinsel ilişki öyküsü",
            "Vajinal akıntı / servisit bulgusu",
        ],
        "Jinekolojik girişim / cihaz": [
            "Rahim içi araç kullanımı",
            "Yakın zamanda rahim içi girişim öyküsü",
            "Histerosalpingografi / histeroskopi / küretaj öyküsü",
        ],
        "Başvuru sırasındaki klinik bulgular": [
            "Ateş",
            "Şiddetli pelvik ağrı",
            "Adneksiyal hassasiyet",
            "Servikal hareket hassasiyeti",
            "Tedaviye yetersiz klinik yanıt",
            "Rüptür şüphesi",
            "Sepsis bulgusu",
        ],
    }


# =====================================================
# PDF
# =====================================================

PDF_FONT_REGULAR = "Helvetica"
PDF_FONT_BOLD = "Helvetica-Bold"


def register_pdf_fonts():
    global PDF_FONT_REGULAR, PDF_FONT_BOLD

    font_candidates = [
        (
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
    ]

    for regular_path, bold_path in font_candidates:
        if os.path.exists(regular_path) and os.path.exists(bold_path):
            try:
                pdfmetrics.registerFont(TTFont("TOARegular", regular_path))
                pdfmetrics.registerFont(TTFont("TOABold", bold_path))
                pdfmetrics.registerFontFamily(
                    "TOAFont",
                    normal="TOARegular",
                    bold="TOABold",
                    italic="TOARegular",
                    boldItalic="TOABold",
                )
                PDF_FONT_REGULAR = "TOARegular"
                PDF_FONT_BOLD = "TOABold"
                break
            except Exception:
                continue


def get_pdf_styles():
    register_pdf_fonts()
    styles = getSampleStyleSheet()

    for style in styles.byName.values():
        style.fontName = PDF_FONT_REGULAR

    styles["Title"].fontName = PDF_FONT_BOLD
    styles["Title"].textColor = colors.HexColor("#1f6fa8")
    styles["Heading1"].fontName = PDF_FONT_BOLD
    styles["Heading1"].textColor = colors.HexColor("#1f6fa8")
    styles["Heading2"].fontName = PDF_FONT_BOLD
    styles["Heading2"].textColor = colors.HexColor("#1f6fa8")
    styles["Heading3"].fontName = PDF_FONT_BOLD
    styles["Heading3"].textColor = colors.HexColor("#1f6fa8")

    return styles


def get_pdf_table_style():
    register_pdf_fonts()
    header_blue = colors.HexColor("#2f80c1")
    border_blue = colors.HexColor("#8abce3")
    row_blue = colors.HexColor("#eef8ff")
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_blue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), row_blue),
        ("GRID", (0, 0), (-1, -1), 0.75, border_blue),
        ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#1f6fa8")),
        ("FONTNAME", (0, 0), (-1, -1), PDF_FONT_REGULAR),
        ("FONTNAME", (0, 0), (-1, 0), PDF_FONT_BOLD),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])


def create_pdf_report(
    patient_info,
    input_values,
    delta_values,
    risk_factors,
    probability,
    prediction_text,
    risk_group,
    model_name
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    styles = get_pdf_styles()
    story = []

    story.append(Paragraph("<b>TOA Operasyon Gereksinimi Tahmin Raporu</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    report_info_table = Table(
        [[f"Rapor tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}"]],
        colWidths=[16 * cm]
    )
    report_info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#dff4ff")),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#8abce3")),
        ("FONTNAME", (0, 0), (-1, -1), PDF_FONT_REGULAR),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#245b82")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(report_info_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Hasta Bilgileri</b>", styles["Heading2"]))

    patient_table_data = [
        ["Alan", "Değer"],
        ["Hasta kodu", str(patient_info.get("Hasta kodu", "-"))],
        ["Yaş", str(patient_info.get("Yaş", "-"))],
        ["Not", str(patient_info.get("Not", "-"))],
    ]

    patient_table = Table(patient_table_data, colWidths=[6 * cm, 10 * cm])
    patient_table.setStyle(get_pdf_table_style())
    story.append(patient_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Girilen Değerler</b>", styles["Heading2"]))

    input_table_data = [["Parametre", "İlk", "Kontrol"]]
    for key, values in input_values.items():
        input_table_data.append([
            key,
            str(values.get("ilk", "-")),
            str(values.get("kontrol", "-"))
        ])

    input_table = Table(input_table_data, colWidths=[6 * cm, 5 * cm, 5 * cm])
    input_table.setStyle(get_pdf_table_style())
    story.append(input_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Delta Değerleri</b>", styles["Heading2"]))
    story.append(Paragraph("Delta hesaplama yönü: İlk değer - kontrol değeri", styles["Normal"]))
    story.append(Spacer(1, 6))

    delta_table_data = [["Değişken", "Değer"]]
    for key, value in delta_values.items():
        delta_table_data.append([key, f"{value:.3f}"])

    delta_table = Table(delta_table_data, colWidths=[8 * cm, 8 * cm])
    delta_table.setStyle(get_pdf_table_style())
    story.append(delta_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>TOA Risk Faktörleri</b>", styles["Heading2"]))

    if risk_factors:
        for rf in risk_factors:
            story.append(Paragraph(f"• {rf}", styles["Normal"]))
    else:
        story.append(Paragraph("Seçilen risk faktörü yok.", styles["Normal"]))

    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Model Sonucu</b>", styles["Heading2"]))

    result_table_data = [
        ["Parametre", "Sonuç"],
        ["Model", model_name],
        ["Operasyon gereksinimi olasılığı", f"{probability:.2%}"],
        ["Model tahmini", prediction_text],
        ["Risk kategorisi", risk_group],
    ]

    result_table = Table(result_table_data, colWidths=[8 * cm, 8 * cm])
    result_table.setStyle(get_pdf_table_style())
    story.append(result_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Uyarı</b>", styles["Heading2"]))
    story.append(Paragraph(
        "Bu rapor araştırma/prototip amaçlıdır. Klinik karar yerine geçmez. "
        "Operasyon kararı hastanın klinik durumu, görüntüleme bulguları, laboratuvar parametreleri, "
        "tedaviye yanıtı ve hekim değerlendirmesi ile birlikte verilmelidir.",
        styles["Normal"]
    ))

    doc.build(story)
    buffer.seek(0)

    return buffer


# =====================================================
# VERİLERİ YÜKLE
# =====================================================

df = load_dataset()
mapping_df = load_mapping()

if df is None:
    st.error(f"`{DATA_FILE}` bulunamadı. Bu dosyayı app.py ile aynı klasöre koymalısın.")
    st.stop()

model_df = prepare_model_df(df)


# =====================================================
# BAŞLIK
# =====================================================

st.sidebar.image("kod.png", use_container_width=True)
st.sidebar.warning("Bu uygulama klinik karar yerine geçmez. Araştırma/prototip amaçlıdır.")

st.title("🩺 TOA Operasyon Gereksinimi Tahmin Prototipi")

st.markdown(
    """
    Bu prototip, anonimleştirilmiş veri seti üzerinden modeli uygulama içinde eğitir.
    Model, **delta WBC, delta NLR, delta CRP ve delta apse boyutu** değişkenleriyle
    operasyon gereksinimi olasılığını hesaplar.
    """
)


# =====================================================
# SEKMELER
# =====================================================

tab_predict, tab_risk, tab_model, tab_report, tab_about = st.tabs([
    "🧮 Risk Hesaplama",
    "⚠️ TOA Risk Faktörleri",
    "📊 Model Performansı",
    "📄 PDF Rapor",
    "ℹ️ Hakkında"
])


# =====================================================

with tab_risk:
    st.header("TOA Risk Faktörleri")

    st.write(
        """
        Bu risk faktörleri modelin matematiksel tahminine dahil edilmez.
        Klinik bağlam ve PDF raporu için kaydedilir.
        """
    )

    risk_factor_options = get_risk_factor_options()
    selected_risk_factors = []

    for category, items in risk_factor_options.items():
        st.subheader(category)

        for item in items:
            checked = st.checkbox(item, key=f"risk_{category}_{item}")
            if checked:
                selected_risk_factors.append(item)

    st.session_state["selected_risk_factors"] = selected_risk_factors

    st.markdown("---")
    st.subheader("Seçilen risk faktörleri")

    if selected_risk_factors:
        for item in selected_risk_factors:
            st.write(f"• {item}")
    else:
        st.write("Henüz risk faktörü seçilmedi.")


# =====================================================
# RİSK HESAPLAMA
# =====================================================

with tab_predict:
    st.header("Operasyon Gereksinimi Tahmini")

    model_name = st.selectbox(
        "Kullanılacak model",
        ["Naive Bayes", "Lojistik Regresyon", "Random Forest", "Gradient Boosting", "SVM"],
        index=0
    )

    final_model, metrics, cm_df, roc_df = train_and_evaluate_model(model_df, model_name)

    st.subheader("Hasta Bilgileri")

    colp1, colp2, colp3 = st.columns(3)

    with colp1:
        patient_code = st.text_input("Hasta kodu / çalışma ID", value="TOA-001")

    with colp2:
        patient_age = st.number_input("Yaş", min_value=0, max_value=100, value=35, step=1)

    with colp3:
        clinical_note = st.text_input("Kısa klinik not", value="")

    delta_values = {
        "var13": wbc_ilk - wbc_kontrol,
        "var16": nlr_ilk - nlr_kontrol,
        "var19": crp_ilk - crp_kontrol,
        "var22": apse_ilk - apse_kontrol,
    }

    X_input = pd.DataFrame([delta_values])[DELTA_FEATURES]

    if st.button("Tahmin Et", type="primary"):
        if hasattr(final_model, "predict_proba"):
            probability = final_model.predict_proba(X_input)[0, 1]
        else:
            prediction_raw = final_model.predict(X_input)[0]
            probability = float(prediction_raw)

        prediction = int(probability >= 0.50)

        if probability < 0.30:
            risk_group = "Düşük risk"
            risk_color = "success"
        elif probability < 0.70:
            risk_group = "Orta risk"
            risk_color = "warning"
        else:
            risk_group = "Yüksek risk"
            risk_color = "error"

        prediction_text = "Operasyon gereksinimi var" if prediction == 1 else "Operasyon gereksinimi yok"

        st.session_state["patient_info"] = {
            "Hasta kodu": patient_code,
            "Yaş": patient_age,
            "Not": clinical_note,
        }

        st.session_state["input_values"] = {
            "WBC": {"ilk": wbc_ilk, "kontrol": wbc_kontrol},
            "NLR": {"ilk": nlr_ilk, "kontrol": nlr_kontrol},
            "CRP": {"ilk": crp_ilk, "kontrol": crp_kontrol},
            "Apse boyutu": {"ilk": apse_ilk, "kontrol": apse_kontrol},
        }

        st.session_state["delta_values"] = {
            "delta WBC": delta_values["var13"],
            "delta NLR": delta_values["var16"],
            "delta CRP": delta_values["var19"],
            "delta apse": delta_values["var22"],
        }

        st.session_state["probability"] = probability
        st.session_state["prediction_text"] = prediction_text
        st.session_state["risk_group"] = risk_group
        st.session_state["model_name"] = model_name

        colr1, colr2, colr3 = st.columns(3)

        with colr1:
            st.metric("Operasyon olasılığı", f"{probability:.2%}")

        with colr2:
            st.metric("Model tahmini", prediction_text)

        with colr3:
            st.metric("Risk kategorisi", risk_group)

        if risk_color == "success":
            st.success("Model düşük operasyon gereksinimi olasılığı hesaplamıştır.")
        elif risk_color == "warning":
            st.warning("Model orta düzey operasyon gereksinimi olasılığı hesaplamıştır.")
        else:
            st.error("Model yüksek operasyon gereksinimi olasılığı hesaplamıştır.")

        st.caption(
            "Bu sonuç klinik karar yerine geçmez; hastanın klinik durumu, görüntüleme bulguları ve tedaviye yanıtı ile birlikte değerlendirilmelidir."
        )


# =====================================================
# MODEL PERFORMANSI
# =====================================================

with tab_model:
    st.header("Model Performansı")

    model_name_perf = st.selectbox(
        "Performansı gösterilecek model",
        ["Naive Bayes", "Lojistik Regresyon", "Random Forest", "Gradient Boosting", "SVM"],
        index=0,
        key="performance_model_select"
    )

    _, metrics_perf, cm_perf, roc_perf = train_and_evaluate_model(model_df, model_name_perf)

    st.subheader(f"{model_name_perf} performans ölçütleri")

    metrics_df = pd.DataFrame([metrics_perf]).T.reset_index()
    metrics_df.columns = ["Metrik", "Değer"]

    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Confusion Matrix")
    st.dataframe(cm_perf, use_container_width=True)

    st.subheader("ROC Grafiği")
    if roc_perf is not None:
        roc_chart_df = roc_perf.set_index("Yanlış pozitif oranı")[["Doğru pozitif oranı"]]
        st.line_chart(roc_chart_df)
        st.caption("ROC eğrisi test verisi üzerinden hesaplanmıştır.")
    else:
        st.warning("ROC grafiği için test kümesinde iki sınıf da bulunmalıdır.")

    st.info(
        "Sınıf dengesizliği varsa Accuracy tek başına yeterli değildir. Balanced Accuracy, AUC ve sınıf bazlı metriklerle birlikte yorumlanmalıdır."
    )


# =====================================================
# PDF RAPOR
# =====================================================

with tab_report:
    st.header("PDF Rapor İndir")

    if "probability" not in st.session_state:
        st.warning("PDF oluşturmak için önce Risk Hesaplama sekmesinde tahmin yapmalısın.")
    else:
        patient_info = st.session_state.get("patient_info", {})
        input_values = st.session_state.get("input_values", {})
        delta_values = st.session_state.get("delta_values", {})
        risk_factors = st.session_state.get("selected_risk_factors", [])
        probability = st.session_state.get("probability", 0.0)
        prediction_text = st.session_state.get("prediction_text", "-")
        risk_group = st.session_state.get("risk_group", "-")
        model_name = st.session_state.get("model_name", "-")

        st.subheader("Rapor önizleme")

        st.write(f"**Model:** {model_name}")
        st.write(f"**Hasta kodu:** {patient_info.get('Hasta kodu', '-')}")
        st.write(f"**Operasyon olasılığı:** {probability:.2%}")
        st.write(f"**Model tahmini:** {prediction_text}")
        st.write(f"**Risk kategorisi:** {risk_group}")

        pdf_buffer = create_pdf_report(
            patient_info=patient_info,
            input_values=input_values,
            delta_values=delta_values,
            risk_factors=risk_factors,
            probability=probability,
            prediction_text=prediction_text,
            risk_group=risk_group,
            model_name=model_name,
        )

        st.download_button(
            label="📄 PDF raporu indir",
            data=pdf_buffer,
            file_name=f"TOA_rapor_{patient_info.get('Hasta kodu', 'hasta')}.pdf",
            mime="application/pdf"
        )


# =====================================================
# HAKKINDA
# =====================================================

with tab_about:
    st.header("Hakkında")

    st.write(
        """
        Bu uygulama, tubo-ovaryan apse tanılı hastalarda operasyon gereksinimini öngörmeye yönelik
        araştırma amaçlı bir klinik karar destek prototipidir.
        """
    )

    st.subheader("Kısıtlılıklar")

    st.write(
        """
        - Model anonimleştirilmiş veri seti üzerinden uygulama içinde eğitilir.
        - Delta değerleri ilk değer - kontrol değeri olarak hesaplanmaktadır.
        - Model yalnızca delta WBC, delta NLR, delta CRP ve delta apse değişkenlerini kullanır.
        - Operasyon gereksinimi olmayan olgu sayısının az olması nedeniyle sonuçlar dikkatli yorumlanmalıdır.
        - Uygulama dış doğrulama yapılmadan klinik karar amacıyla kullanılmamalıdır.
        """
    )