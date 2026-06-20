import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

from imblearn.over_sampling import SMOTE

try:
    import shap
except Exception:
    shap = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


st.set_page_config(
    page_title="Şiddetli Preeklampsi Risk Hesaplayıcı",
    page_icon="🩺",
    layout="wide",
)

# ── Değişen kısım: sütun adları güncellendi ──────────────────────────────────
FEATURES = ["f_platelet", "f_inr", "f_aptt", "f_fibrinogen"]
TARGET = "target"
MODEL_VERSION = "v1.1"
FEATURE_LABELS = {
    "f_platelet": "Trombosit",
    "f_inr": "INR",
    "f_aptt": "aPTT",
    "f_fibrinogen": "Fibrinojen",
}
# ─────────────────────────────────────────────────────────────────────────────

ACOG_CHECKLIST = [
    ("acog_bp", "Kan Basıncı ≥160/110 mmHg"),
    ("acog_plt_low", "Trombosit <100,000/µL"),
    ("acog_cr", "Serum Kreatin >1.1 mg/dL"),
    ("acog_liver", "Karaciğer enzim yüksekliği / Şiddetli ağrı"),
    ("acog_pulmonary", "Pulmoner ödem"),
    ("acog_neuro", "Nörolojik semptomlar"),
]

DATA_DICTIONARY = [
    ("f_platelet", "Trombosit", "/µL", "Koagülasyon ve hematolojik durum için kullanılan trombosit sayısı."),
    ("f_inr", "INR", "oran", "Protrombin zamanı temelli pıhtılaşma göstergesi."),
    ("f_aptt", "aPTT", "saniye", "İntrinsik koagülasyon yolunu yansıtan süre ölçümü."),
    ("f_fibrinogen", "Fibrinojen", "mg/dL", "Koagülasyon sisteminde yer alan plazma proteini."),
    (TARGET, "Şiddetli preeklampsi", "0/1", "Modelin tahmin etmeye çalıştığı hedef değişken."),
]

LIMITATIONS = [
    "Bu araç araştırma prototipidir ve klinik karar yerine geçmez.",
    "Modelin güvenilirliği eğitim verisinin temsil gücü ve örneklem büyüklüğü ile sınırlıdır.",
    "Harici doğrulama yapılmadan farklı merkez, popülasyon veya cihaz ölçümlerine genellenmemelidir.",
    "ACOG bulguları ve klinik değerlendirme, model çıktısının önünde yorumlanmalıdır.",
]

LIMITATIONS_EN = [
    "This tool is a research prototype and does not replace clinical decision-making.",
    "Model reliability is limited by the representativeness and size of the training data.",
    "It should not be generalised to different centres, populations or measurement devices without external validation.",
    "ACOG findings and clinical assessment should take precedence over model output.",
]

# ── Bilingual text dictionary ──────────────────────────────────────────────
T = {
    "tr": {
        "page_title": "Şiddetli Preeklampsi Risk Hesaplayıcı",
        "page_caption": "Açıklanabilir, raporlanabilir ve araştırma odaklı Streamlit prototipi",
        "sidebar_info": "SMOTE ile dengelenmiş lojistik regresyon modeli kullanarak şiddetli preeklampsi riskini tahmin eder.",
        "tabs": ["Risk Hesaplayıcı", "Klinik Özet", "ACOG", "Açıklanabilirlik",
                 "Toplu Tahmin", "Model Karşılaştırma", "Kalibrasyon ve Fayda",
                 "Veri ve Trend", "PDF Raporu", "Metodoloji"],
        "patient_info": "Hasta Bilgileri",
        "patient_id": "Hasta ID / Protokol No",
        "age": "Yaş",
        "gest_week": "Gebelik haftası",
        "gravida": "Gravida / Parite",
        "lab_values": "Laboratuvar Değerleri",
        "threshold_opt": "Eşik Optimizasyonu",
        "threshold_strategies": ["Dengeli eşik", "Duyarlılığı önceliklendir", "Özgüllüğü önceliklendir", "Sabit 0.50"],
        "threshold_caption": "Seçilen strateji eşiği",
        "risk_summary_radio": ["Risk Özeti", "Güven Aralığı", "Aralık Kontrolü"],
        "bootstrap_btn": "📊 Detaylı Güven Aralığı Analizi Yap (80 İterasyon)",
        "bootstrap_wait": "Bootstrap hesaplanıyor, lütfen bekleyin...",
        "bootstrap_result": "Bootstrap belirsizlik aralığı",
        "bootstrap_start_info": "Güven aralığını görmek için analizi başlatın.",
        "range_control": "Eğitim Verisi Aralığı Kontrolü",
        "range_warning": "Veri aralığı uyarısı",
        "risk_ratio": "Risk Oranı",
        "category_label": "Kategori",
        "threshold_label": "Eşik",
        "clinical_summary_title": "Kombine Klinik Özet",
        "risk_category_label": "Risk kategorisi",
        "model_label": "Model",
        "acog_label": "ACOG",
        "risk_label": "Risk",
        "high_risk": "Yüksek Risk",
        "low_risk": "Düşük Risk",
        "acog_positive": "Şiddetli özellik mevcut",
        "acog_negative": "Seçili şiddetli özellik yok",
        "acog_title": "ACOG Şiddetli Özellikler Kontrol Listesi",
        "acog_bp": "Kan Basıncı ≥160/110 mmHg",
        "acog_plt": "Trombosit <100,000/µL",
        "acog_cr": "Serum Kreatin >1.1 mg/dL",
        "acog_liver": "Karaciğer enzim yüksekliği / Şiddetli ağrı",
        "acog_pulm": "Pulmoner ödem",
        "acog_neuro": "Nörolojik semptomlar",
        "acog_alert": "En az bir ACOG şiddetli özelliği mevcut.",
        "acog_ok": "Seçili ACOG şiddetli özelliği bulunmuyor.",
        "shap_title": "Yapay Zeka Açıklanabilirlik",
        "shap_btn": "🔍 Karar Analizini Başlat (SHAP)",
        "shap_spinner": "Hesaplanıyor...",
        "shap_no_pkg": "SHAP grafikleri için `shap` paketi yüklü olmalı. Kurulum: `pip install shap`",
        "shap_info": "Analizi görmek için yukarıdaki butona tıklayın. Bu işlem işlemciyi yorabilir.",
        "bulk_title": "Çoklu Hasta Tahmini",
        "bulk_caption": "CSV veya Excel dosyasında şu sütunlar bulunmalı: f_platelet, f_inr, f_aptt, f_fibrinogen",
        "bulk_upload": "Hasta veri dosyası yükle",
        "bulk_csv_btn": "Sonuçları CSV olarak indir",
        "bulk_excel_btn": "Sonuçları Excel olarak indir",
        "bulk_excel_info": "Excel çıktısı için `openpyxl` paketi gerekir. CSV çıktısı hazır.",
        "bulk_error": "Toplu tahmin yapılamadı",
        "compare_title": "Model Karşılaştırma",
        "compare_btn": "Model Karşılaştırmayı Başlat",
        "compare_spinner": "Modeller karşılaştırılıyor...",
        "compare_info": "Karşılaştırmayı görmek için yukarıdaki butona tıklayın.",
        "xgb_info": "XGBoost yüklü değilse karşılaştırmaya dahil edilmez. Kurulum: `pip install xgboost`",
        "calib_title": "Kalibrasyon Grafiği",
        "calib_xlabel": "Ortalama tahmin edilen risk",
        "calib_ylabel": "Gözlenen pozitif oran",
        "calib_model": "Model",
        "calib_ideal": "İdeal",
        "dca_title": "Decision Curve Analysis",
        "dca_xlabel": "Eşik olasılığı",
        "dca_ylabel": "Net benefit",
        "dca_model": "Model",
        "dca_treat_all": "Herkesi yüksek risk kabul et",
        "dca_treat_none": "Hiçbirini yüksek risk kabul et",
        "data_title": "Veri Seti Analiz Paneli",
        "data_total": "Toplam satır",
        "data_model": "Model satırı",
        "data_pos": "Pozitif oran",
        "data_missing": "Eksik değer",
        "data_dist": "Sınıf dağılımı",
        "data_var_dist": "Değişken dağılımları",
        "data_select": "Değişken",
        "data_freq": "Frekans",
        "trend_title": "Hasta Trend Takibi",
        "trend_upload": "Aynı hastaya ait seri ölçümler yükle",
        "trend_warning": "Trend dosyasında en az bir laboratuvar sütunu bulunmalı.",
        "trend_error": "Trend dosyası okunamadı",
        "trend_time": "Zaman",
        "trend_value": "Değer",
        "pdf_title": "PDF Raporu",
        "pdf_model_perf": "Model Performansı",
        "pdf_include_shap": "PDF'ye SHAP grafiğini ekle",
        "pdf_create_btn": "Raporu Oluştur",
        "pdf_download_btn": "PDF Dosyasını İndir",
        "method_title": "Metodoloji",
        "method_text": ("Bu araştırma prototipi platelet, INR, aPTT ve fibrinojen değerlerini kullanarak şiddetli preeklampsi riskini tahmin eder. "
                        "Birincil model lojistik regresyondur. Eğitim setindeki sınıf dengesizliğini azaltmak için SMOTE uygulanır; "
                        "performans metrikleri ayrılmış test seti üzerinde hesaplanır."),
        "dict_title": "Veri Sözlüğü",
        "dict_cols": ["Değişken", "Klinik ad", "Birim", "Açıklama"],
        "limits_title": "Sınırlılıklar",
        "limitations": LIMITATIONS,
        "stamp_title": "Model ve Veri Sürümü",
        "stamp_cols": ["Alan", "Değer"],
        "ref_title": "Eğitim Verisi Referans Aralıkları",
    },
    "en": {
        "page_title": "Severe Preeclampsia Risk Calculator",
        "page_caption": "Explainable, reportable, research-focused Streamlit prototype",
        "sidebar_info": "Estimates severe preeclampsia risk using a SMOTE-balanced logistic regression model.",
        "tabs": ["Risk Calculator", "Clinical Summary", "ACOG", "Explainability",
                 "Batch Prediction", "Model Comparison", "Calibration & Utility",
                 "Data & Trends", "PDF Report", "Methodology"],
        "patient_info": "Patient Information",
        "patient_id": "Patient ID / Protocol No",
        "age": "Age",
        "gest_week": "Gestational week",
        "gravida": "Gravida / Parity",
        "lab_values": "Laboratory Values",
        "threshold_opt": "Threshold Optimisation",
        "threshold_strategies": ["Balanced threshold", "Prioritise sensitivity", "Prioritise specificity", "Fixed 0.50"],
        "threshold_caption": "Selected strategy threshold",
        "risk_summary_radio": ["Risk Summary", "Confidence Interval", "Range Check"],
        "bootstrap_btn": "📊 Run Detailed Confidence Interval Analysis (80 iterations)",
        "bootstrap_wait": "Calculating bootstrap, please wait...",
        "bootstrap_result": "Bootstrap uncertainty interval",
        "bootstrap_start_info": "Start the analysis to see the confidence interval.",
        "range_control": "Training Data Range Check",
        "range_warning": "Data range warning",
        "risk_ratio": "Risk Score",
        "category_label": "Category",
        "threshold_label": "Threshold",
        "clinical_summary_title": "Combined Clinical Summary",
        "risk_category_label": "Risk category",
        "model_label": "Model",
        "acog_label": "ACOG",
        "risk_label": "Risk",
        "high_risk": "High Risk",
        "low_risk": "Low Risk",
        "acog_positive": "Severe feature present",
        "acog_negative": "No selected severe feature",
        "acog_title": "ACOG Severe Features Checklist",
        "acog_bp": "Blood Pressure ≥160/110 mmHg",
        "acog_plt": "Platelet <100,000/µL",
        "acog_cr": "Serum Creatinine >1.1 mg/dL",
        "acog_liver": "Elevated liver enzymes / Severe pain",
        "acog_pulm": "Pulmonary oedema",
        "acog_neuro": "Neurological symptoms",
        "acog_alert": "At least one ACOG severe feature is present.",
        "acog_ok": "No selected ACOG severe feature.",
        "shap_title": "AI Explainability",
        "shap_btn": "🔍 Start Decision Analysis (SHAP)",
        "shap_spinner": "Calculating...",
        "shap_no_pkg": "The `shap` package is required for SHAP charts. Install: `pip install shap`",
        "shap_info": "Click the button above to see the analysis. This may take a moment.",
        "bulk_title": "Batch Patient Prediction",
        "bulk_caption": "CSV or Excel file must contain columns: f_platelet, f_inr, f_aptt, f_fibrinogen",
        "bulk_upload": "Upload patient data file",
        "bulk_csv_btn": "Download results as CSV",
        "bulk_excel_btn": "Download results as Excel",
        "bulk_excel_info": "`openpyxl` is required for Excel output. CSV is ready.",
        "bulk_error": "Batch prediction failed",
        "compare_title": "Model Comparison",
        "compare_btn": "Start Model Comparison",
        "compare_spinner": "Comparing models...",
        "compare_info": "Click the button above to see the comparison.",
        "xgb_info": "XGBoost will not be included if not installed. Install: `pip install xgboost`",
        "calib_title": "Calibration Plot",
        "calib_xlabel": "Mean predicted risk",
        "calib_ylabel": "Observed positive rate",
        "calib_model": "Model",
        "calib_ideal": "Ideal",
        "dca_title": "Decision Curve Analysis",
        "dca_xlabel": "Threshold probability",
        "dca_ylabel": "Net benefit",
        "dca_model": "Model",
        "dca_treat_all": "Treat all as high risk",
        "dca_treat_none": "Treat none as high risk",
        "data_title": "Dataset Analysis Panel",
        "data_total": "Total rows",
        "data_model": "Model rows",
        "data_pos": "Positive rate",
        "data_missing": "Missing values",
        "data_dist": "Class distribution",
        "data_var_dist": "Variable distributions",
        "data_select": "Variable",
        "data_freq": "Frequency",
        "trend_title": "Patient Trend Tracking",
        "trend_upload": "Upload serial measurements for the same patient",
        "trend_warning": "Trend file must contain at least one laboratory column.",
        "trend_error": "Could not read trend file",
        "trend_time": "Time",
        "trend_value": "Value",
        "pdf_title": "PDF Report",
        "pdf_model_perf": "Model Performance",
        "pdf_include_shap": "Include SHAP chart in PDF",
        "pdf_create_btn": "Generate Report",
        "pdf_download_btn": "Download PDF File",
        "method_title": "Methodology",
        "method_text": ("This research prototype estimates severe preeclampsia risk using platelet, INR, aPTT and fibrinogen values. "
                        "The primary model is logistic regression. SMOTE is applied to the training split to reduce class imbalance, "
                        "and performance metrics are calculated on a held-out test split."),
        "dict_title": "Data Dictionary",
        "dict_cols": ["Variable", "Clinical label", "Unit", "Description"],
        "limits_title": "Limitations",
        "limitations": LIMITATIONS_EN,
        "stamp_title": "Model & Data Version",
        "stamp_cols": ["Field", "Value"],
        "ref_title": "Training Data Reference Ranges",
    },
}


def register_pdf_fonts():
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("AppFont", str(regular)))
        pdfmetrics.registerFont(TTFont("AppFont-Bold", str(bold)))
        return "AppFont", "AppFont-Bold"
    return "Helvetica", "Helvetica-Bold"


def clean_numeric_dataframe(df):
    out = df.copy()
    out.columns = out.columns.str.strip()
    out = out.replace(",", ".", regex=True)
    for col in out.columns:
        try:
            out[col] = pd.to_numeric(out[col])
        except (TypeError, ValueError):
            pass
    return out


@st.cache_data(show_spinner=False)
def load_training_dataframe(csv_path, csv_mtime):
    return clean_numeric_dataframe(pd.read_csv(csv_path))


def dataframe_signature(df):
    hashed = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64)
    return int(hashed.sum(dtype=np.uint64))


def feature_label(feature):
    return FEATURE_LABELS.get(feature, feature)


def current_acog_items():
    return {label: bool(st.session_state.get(key, False)) for key, label in ACOG_CHECKLIST}


def get_model_stamp():
    return {
        "Model versiyonu": MODEL_VERSION,
        "Rapor tarihi": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "Veri dosyası": "tezveri.csv",
    }


def feature_range_summary(reference_data):
    rows = []
    for feature in FEATURES:
        series = pd.to_numeric(reference_data[feature], errors="coerce").dropna()
        rows.append(
            {
                "Değişken": feature_label(feature),
                "Minimum": float(series.min()),
                "1. persentil": float(series.quantile(0.01)),
                "Medyan": float(series.median()),
                "99. persentil": float(series.quantile(0.99)),
                "Maksimum": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def patient_range_warnings(patient_values, reference_data):
    rows = []
    messages = []
    for feature in FEATURES:
        value = float(patient_values[feature])
        series = pd.to_numeric(reference_data[feature], errors="coerce").dropna()
        min_val = float(series.min())
        max_val = float(series.max())
        p01 = float(series.quantile(0.01))
        p99 = float(series.quantile(0.99))
        if value < min_val or value > max_val:
            status = "Eğitim aralığı dışında"
            messages.append(f"{feature_label(feature)} değeri eğitim verisindeki min-maks aralığının dışında.")
        elif value < p01 or value > p99:
            status = "Uç sınıra yakın"
            messages.append(f"{feature_label(feature)} değeri eğitim verisinin uç persentil aralığında.")
        else:
            status = "Aralık içinde"
        rows.append(
            {
                "Parametre": feature_label(feature),
                "Hasta değeri": value,
                "Eğitim min": min_val,
                "Eğitim max": max_val,
                "Durum": status,
            }
        )
    return pd.DataFrame(rows), messages


@st.cache_data(show_spinner=False)
def bootstrap_patient_risk_ci(patient_values, X_train_df, y_train_series, n_bootstraps=80):
    rng = np.random.default_rng(42)
    X_boot_source = X_train_df.reset_index(drop=True)
    y_boot_source = y_train_series.reset_index(drop=True).astype(int)
    patient_df = pd.DataFrame([patient_values], columns=FEATURES)
    risks = []

    for _ in range(n_bootstraps):
        sample_idx = rng.integers(0, len(X_boot_source), len(X_boot_source))
        X_sample = X_boot_source.iloc[sample_idx]
        y_sample = y_boot_source.iloc[sample_idx]
        class_counts = y_sample.value_counts()
        if len(class_counts) < 2:
            continue

        try:
            if class_counts.min() > 1:
                k_neighbors = min(5, int(class_counts.min()) - 1)
                sampler = SMOTE(random_state=int(rng.integers(0, 1_000_000)), k_neighbors=k_neighbors)
                X_resampled, y_resampled = sampler.fit_resample(X_sample, y_sample)
            else:
                X_resampled, y_resampled = X_sample, y_sample

            boot_scaler = StandardScaler()
            X_scaled = boot_scaler.fit_transform(X_resampled)
            patient_scaled = boot_scaler.transform(patient_df)
            boot_model = LogisticRegression(random_state=42, max_iter=1000)
            boot_model.fit(X_scaled, y_resampled)
            risks.append(float(boot_model.predict_proba(patient_scaled)[0, 1]))
        except Exception:
            continue

    if len(risks) < 10:
        return None
    low, high = np.percentile(risks, [2.5, 97.5])
    return float(low), float(high)


def risk_category(risk, pred_class, acog_any):
    if acog_any and pred_class == 1:
        return "Kritik klinik uyarı", "error"
    if acog_any:
        return "Klinik uyarı", "warning"
    if risk >= 0.75:
        return "Yüksek risk", "error"
    if risk >= 0.40:
        return "Orta risk", "warning"
    return "Düşük risk", "success"


def clinical_summary(risk, threshold, pred_class, acog_any, category):
    if category == "Kritik klinik uyarı":
        return "Model yüksek risk öngörüyor ve hastada ACOG şiddetli özelliği mevcut. Klinik bulgular öncelikli değerlendirilmelidir."
    if pred_class == 1:
        return "Model yüksek risk öngörüyor. Laboratuvar değerleri, hasta semptomları ve izlem trendi birlikte yorumlanmalıdır."
    if acog_any:
        return "Model düşük risk öngörüyor ancak ACOG şiddetli özelliği mevcut. Klinik bulgu model çıktısının önünde değerlendirilmelidir."
    if risk >= threshold * 0.75:
        return "Model düşük risk sınıfında ancak eşik değerine yakın. Yakın izlem ve trend değerlendirmesi yararlı olabilir."
    return "Model düşük risk öngörüyor ve seçili ACOG şiddetli özelliği bulunmuyor. Sonuç hasta bağlamıyla birlikte değerlendirilmelidir."


def threshold_from_strategy(strategy, thresholds, fpr, tpr):
    if strategy == "Dengeli eşik":
        return float(thresholds[int(np.argmax(tpr - fpr))])
    if strategy == "Duyarlılığı önceliklendir":
        candidates = np.where(tpr >= 0.90)[0]
        return float(thresholds[candidates[-1]]) if len(candidates) else float(thresholds[int(np.argmax(tpr - fpr))])
    if strategy == "Özgüllüğü önceliklendir":
        specificity = 1 - fpr
        candidates = np.where(specificity >= 0.90)[0]
        return float(thresholds[candidates[0]]) if len(candidates) else float(thresholds[int(np.argmax(tpr - fpr))])
    return 0.50


def evaluate_at_threshold(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0,
        "ppv": tp / (tp + fp) if (tp + fp) else 0,
        "npv": tn / (tn + fn) if (tn + fn) else 0,
        "pred": y_pred,
        "cm": np.array([[tn, fp], [fn, tp]]),
    }


@st.cache_resource(show_spinner=False)
def train_primary_model(data_signature_value, _data):
    X = _data[FEATURES]
    y = _data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train_smote)
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
    auc = roc_auc_score(y_test, y_prob_test)
    default_threshold = float(thresholds[int(np.argmax(tpr - fpr))])
    default_vals = {feature: float(_data[feature].median()) for feature in FEATURES}

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "model": model,
        "y_prob_test": y_prob_test,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc,
        "default_threshold": default_threshold,
        "default_vals": default_vals,
    }


@st.cache_resource(show_spinner=False)
def train_candidate_models(data_signature_value, _X_train_scaled, _y_train_smote, _X_test_scaled, _y_test):
    candidates = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    }
    if XGBClassifier is not None:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    rows = []
    fitted = {}
    for name, candidate in candidates.items():
        candidate.fit(_X_train_scaled, _y_train_smote)
        prob = candidate.predict_proba(_X_test_scaled)[:, 1]
        fpr_model, tpr_model, th_model = roc_curve(_y_test, prob)
        threshold = float(th_model[int(np.argmax(tpr_model - fpr_model))])
        metrics = evaluate_at_threshold(_y_test, prob, threshold)
        rows.append(
            {
                "Model": name,
                "AUC": roc_auc_score(_y_test, prob),
                "Accuracy": metrics["accuracy"],
                "Sensitivity": metrics["sensitivity"],
                "Specificity": metrics["specificity"],
                "Optimal Threshold": threshold,
            }
        )
        fitted[name] = candidate
    return pd.DataFrame(rows).sort_values("AUC", ascending=False), fitted


@st.cache_resource(show_spinner=False)
def shap_dataframe(data_signature_value, _model, _train_scaled, patient_scaled_values, patient_values):
    if shap is None:
        return None, None
    patient_scaled = np.array([patient_scaled_values], dtype=float)
    explainer = shap.LinearExplainer(_model, _train_scaled, feature_perturbation="interventional")
    shap_values = np.array(explainer.shap_values(patient_scaled)).reshape(-1)
    df = pd.DataFrame(
        {
            "Parametre": [feature_label(f) for f in FEATURES],
            "Hasta Değeri": list(patient_values),
            "SHAP Etkisi": shap_values,
            "Mutlak Etki": np.abs(shap_values),
        }
    ).sort_values("Mutlak Etki", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = ["#d9534f" if val > 0 else "#2e8b57" for val in df["SHAP Etkisi"]]
    ax.barh(df["Parametre"], df["SHAP Etkisi"], color=bar_colors)
    ax.axvline(0, color="#202124", linewidth=1)
    ax.set_xlabel("Modele etki (pozitif değer riski artırır)")
    ax.invert_yaxis()
    return df, fig


def fig_to_png_buffer(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=180)
    buffer.seek(0)
    return buffer


def predict_patients(input_df, scaler, model, threshold):
    missing = [feature for feature in FEATURES if feature not in input_df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {', '.join(missing)}")

    result = input_df.copy()
    patient_features = result[FEATURES].replace(",", ".", regex=True)
    for col in FEATURES:
        patient_features[col] = pd.to_numeric(patient_features[col], errors="coerce")

    valid_mask = patient_features.notna().all(axis=1)
    risks = np.full(len(result), np.nan)
    preds = np.full(len(result), np.nan)
    if valid_mask.any():
        valid_scaled = scaler.transform(patient_features.loc[valid_mask, FEATURES])
        valid_risks = model.predict_proba(valid_scaled)[:, 1]
        risks[valid_mask.to_numpy()] = valid_risks
        preds[valid_mask.to_numpy()] = (valid_risks >= threshold).astype(int)

    result["risk_probability"] = risks
    result["risk_percent"] = np.round(risks * 100, 1)
    result["risk_class"] = np.where(preds == 1, "Yüksek Risk", np.where(preds == 0, "Düşük Risk", "Hesaplanamadı"))
    return result


def dataframe_to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Risk Tahminleri")
    buffer.seek(0)
    return buffer


def pdf_safe_text(value):
    text = "" if value is None else str(value)
    replacements = {
        "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G",
        "ı": "i", "İ": "I", "ö": "o", "Ö": "O",
        "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
        "µ": "u", "•": "-", "≥": ">=",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "ignore").decode("ascii")


def pdf_safe_rows(rows):
    return [[pdf_safe_text(cell) for cell in row] for row in rows]


def create_pdf_report(
    patient_info, labs, risk, threshold, category, acog_pos, checklist,
    metrics, summary, shap_fig=None, risk_ci=None, range_warnings=None, model_stamp=None,
):
    buffer = BytesIO()
    font_regular, font_bold = register_pdf_fonts()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.6 * cm, leftMargin=1.6 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="AppTitle", fontName=font_bold, fontSize=17, leading=22, textColor=colors.HexColor("#6b3fa0")))
    styles.add(ParagraphStyle(name="AppHeading", fontName=font_bold, fontSize=12, leading=16, textColor=colors.HexColor("#6b3fa0")))
    styles.add(ParagraphStyle(name="AppBody", fontName=font_regular, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="AppSmall", fontName=font_regular, fontSize=8, leading=10, textColor=colors.HexColor("#555555")))

    story = [
        Paragraph(pdf_safe_text("Siddetli Preeklampsi Risk Raporu"), styles["AppTitle"]),
        Spacer(1, 8),
        Paragraph(pdf_safe_text("Bu rapor arastirma prototipi ciktisidir; klinik karar yerine gecmez."), styles["AppSmall"]),
        Spacer(1, 12),
    ]

    def styled_table(rows, widths):
        table = Table(pdf_safe_rows(rows), colWidths=widths)
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_regular),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6b3fa0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d4c6ee")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    story.append(Paragraph(pdf_safe_text("Hasta Bilgileri"), styles["AppHeading"]))
    story.append(styled_table([["Alan", "Deger"]] + [[k, v or "-"] for k, v in patient_info.items()], [6 * cm, 9 * cm]))

    story.extend([Spacer(1, 10), Paragraph(pdf_safe_text("Klinik Ozet"), styles["AppHeading"])])
    summary_rows = [
        ["Alan", "Deger"],
        ["Risk Orani", f"%{risk * 100:.1f}"],
        ["Risk Esigi", f"%{threshold * 100:.1f}"],
        ["Risk Kategorisi", category],
        ["ACOG Siddetli Ozellik", "Var" if acog_pos else "Yok"],
    ]
    if risk_ci is not None:
        summary_rows.insert(2, ["Bootstrap belirsizlik araligi", f"%{risk_ci[0] * 100:.1f} - %{risk_ci[1] * 100:.1f}"])
    story.append(styled_table(summary_rows, [6 * cm, 9 * cm]))
    story.extend([Spacer(1, 8), Paragraph(pdf_safe_text(summary), styles["AppBody"]), Spacer(1, 10)])

    if range_warnings:
        story.append(Paragraph(pdf_safe_text("Veri Araligi Uyarilari"), styles["AppHeading"]))
        story.append(styled_table([["Uyari"]] + [[warning] for warning in range_warnings], [15 * cm]))
        story.append(Spacer(1, 10))

    story.append(Paragraph(pdf_safe_text("Laboratuvar Degerleri"), styles["AppHeading"]))
    story.append(styled_table([["Parametre", "Deger"]] + [[feature_label(k), f"{v:g}"] for k, v in labs.items()], [6 * cm, 9 * cm]))

    story.extend([Spacer(1, 10), Paragraph(pdf_safe_text("ACOG Kontrol Listesi"), styles["AppHeading"])])
    story.append(styled_table([["ACOG Maddesi", "Durum"]] + [[k, "Pozitif" if v else "Negatif"] for k, v in checklist.items()], [10 * cm, 5 * cm]))

    story.extend([Spacer(1, 10), Paragraph(pdf_safe_text("Model Performansi"), styles["AppHeading"])])
    story.append(styled_table([["Metrik", "Deger"]] + [[k, f"{v:.3f}"] for k, v in metrics.items()], [6 * cm, 9 * cm]))

    if shap_fig is not None:
        story.extend([Spacer(1, 12), Paragraph(pdf_safe_text("SHAP Aciklanabilirlik Grafigi"), styles["AppHeading"])])
        shap_buffer = fig_to_png_buffer(shap_fig)
        story.append(Image(shap_buffer, width=15 * cm, height=7.5 * cm))

    if model_stamp:
        story.extend([Spacer(1, 12), Paragraph(pdf_safe_text("Model ve Rapor Bilgisi"), styles["AppHeading"])])
        story.append(styled_table([["Alan", "Deger"]] + [[k, v] for k, v in model_stamp.items()], [6 * cm, 9 * cm]))

    story.extend([Spacer(1, 12), Paragraph(pdf_safe_text("Metodoloji ve Sinirliliklar"), styles["AppHeading"])])
    method_text = (
        "Model, platelet, INR, aPTT ve fibrinojen degiskenleriyle lojistik regresyon kullanir. "
        "Egitim verisinde sinif dengesizligini azaltmak icin SMOTE uygulanir; performans train-test ayrimi uzerinde raporlanir."
    )
    story.append(Paragraph(pdf_safe_text(method_text), styles["AppBody"]))
    for limitation in LIMITATIONS:
        story.append(Paragraph(pdf_safe_text(f"- {limitation}"), styles["AppSmall"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


st.markdown(
    """
<style>
.stApp { background-color: #f7f4fb; }
h1, h2, h3 { color: #6b3fa0; }
[data-testid="stMetricValue"] { color: #6b3fa0; }
.stButton > button, .stDownloadButton > button {
    background-color: #6b3fa0; color: white; border-radius: 8px; border: none; width: 100%;
}
.stButton > button:hover, .stDownloadButton > button:hover { background-color: #563080; color: white; }
.clinical-box {
    padding: 18px 20px; border-radius: 8px; border: 1px solid #d4c6ee;
    background: #ffffff; font-size: 1rem; line-height: 1.5;
}
.mulan-warning-card {
    display: flex; gap: 12px; align-items: flex-start;
    background: #fff9d9; color: #8a6a00; border-radius: 8px;
    padding: 16px 18px; font-size: 0.98rem; line-height: 1.45;
}
.mulan-warning-icon { font-size: 1.25rem; line-height: 1.4; }
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    try:
        st.image("mulan.png", use_container_width=True)
    except Exception:
        st.warning("'mulan.png' dosyası bulunamadı.")
    st.markdown("<h2 style='text-align: center; color: #6b3fa0;'>powered by mulan 🐾</h2>", unsafe_allow_html=True)
    st.markdown("---")
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    lang = "en" if language == "English" else "tr"
    st.session_state["language"] = language
    st.session_state["lang"] = lang
    st.caption(f"Model {MODEL_VERSION}")
    st.info(T[lang]["sidebar_info"])
    st.markdown(
        """
<div class="mulan-warning-card">
  <span class="mulan-warning-icon">⚠️</span>
  <span>Sadece araştırma amaçlıdır.<br>Klinik kararlar için kullanılmaz.</span>
</div>
""",
        unsafe_allow_html=True,
    )

lang = st.session_state.get("lang", "tr")

st.title("🩺 " + T[lang]["page_title"])
st.caption(T[lang]["page_caption"])


try:
    data_path = Path("tezveri.csv")
    df = load_training_dataframe(str(data_path), data_path.stat().st_mtime)
    if TARGET not in df.columns or any(feature not in df.columns for feature in FEATURES):
        st.error("Gerekli sütunlar CSV dosyasında bulunamadı.")
        st.stop()

    data = df[FEATURES + [TARGET]].dropna()
    data_sig = dataframe_signature(data)
    model_artifacts = train_primary_model(data_sig, data)
    X = model_artifacts["X"]
    y = model_artifacts["y"]
    X_train = model_artifacts["X_train"]
    X_test = model_artifacts["X_test"]
    y_train = model_artifacts["y_train"]
    y_test = model_artifacts["y_test"]
    X_train_smote = model_artifacts["X_train_smote"]
    y_train_smote = model_artifacts["y_train_smote"]
    X_train_scaled = model_artifacts["X_train_scaled"]
    X_test_scaled = model_artifacts["X_test_scaled"]
    scaler = model_artifacts["scaler"]
    model = model_artifacts["model"]
    y_prob_test = model_artifacts["y_prob_test"]
    fpr = model_artifacts["fpr"]
    tpr = model_artifacts["tpr"]
    thresholds = model_artifacts["thresholds"]
    auc = model_artifacts["auc"]
    default_threshold = model_artifacts["default_threshold"]
    default_vals = model_artifacts["default_vals"]
except Exception as e:
    st.error(f"Veri yüklenirken hata oluştu: {e}")
    st.stop()


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(T[lang]["tabs"])


with tab1:
    st.subheader(T[lang]["patient_info"])
    info_cols = st.columns(4)
    patient_id = info_cols[0].text_input(T[lang]["patient_id"])
    patient_age = info_cols[1].number_input(T[lang]["age"], min_value=10, max_value=60, value=30)
    gest_week = info_cols[2].number_input(T[lang]["gest_week"], min_value=10.0, max_value=42.0, value=32.0, step=0.1)
    gravida_parity = info_cols[3].text_input(T[lang]["gravida"])

    st.subheader(T[lang]["lab_values"])
    col_l, col_r = st.columns(2)
    with col_l:
        platelet = st.number_input("Trombosit / Platelet", value=default_vals["f_platelet"])
        inr = st.number_input("INR", value=default_vals["f_inr"])
    with col_r:
        aptt = st.number_input("aPTT", value=default_vals["f_aptt"])
        fibrinogen = st.number_input("Fibrinojen / Fibrinogen", value=default_vals["f_fibrinogen"])

    strategy = st.selectbox(T[lang]["threshold_opt"], T[lang]["threshold_strategies"])
    strategy_map = {v: k for k, v in zip(
        ["Dengeli eşik", "Duyarlılığı önceliklendir", "Özgüllüğü önceliklendir", "Sabit 0.50"],
        T[lang]["threshold_strategies"]
    )}
    strategy_tr = strategy_map.get(strategy, strategy)
    recommended_threshold = threshold_from_strategy(strategy_tr, thresholds, fpr, tpr)
    risk_threshold = recommended_threshold
    st.caption(f"{T[lang]['threshold_caption']}: %{risk_threshold * 100:.1f}")

    patient = pd.DataFrame([[platelet, inr, aptt, fibrinogen]], columns=FEATURES)
    patient_scaled = scaler.transform(patient)
    risk = model.predict_proba(patient_scaled)[0, 1]
    pred_class = 1 if risk >= risk_threshold else 0
    current_acog = any(current_acog_items().values())
    category, category_level = risk_category(risk, pred_class, current_acog)
    summary_text = clinical_summary(risk, risk_threshold, pred_class, current_acog, category)
    threshold_metrics = evaluate_at_threshold(y_test, y_prob_test, risk_threshold)
    range_df, range_messages = patient_range_warnings(patient.iloc[0], data)
    patient_values = tuple(patient.iloc[0].to_list())
    risk_ci_key = (patient_values, data_sig)
    if st.session_state.get("risk_ci_key") != risk_ci_key:
        st.session_state["risk_ci"] = None
        st.session_state["risk_ci_key"] = risk_ci_key
    risk_ci = st.session_state.get("risk_ci", None)

    st.session_state.update({
        "patient_info": {
            T[lang]["patient_id"]: patient_id,
            T[lang]["age"]: str(patient_age),
            T[lang]["gest_week"]: str(gest_week),
            T[lang]["gravida"]: gravida_parity,
        },
        "labs": {"f_platelet": platelet, "f_inr": inr, "f_aptt": aptt, "f_fibrinogen": fibrinogen},
        "risk": risk,
        "pred": pred_class,
        "category": category,
        "thresh": risk_threshold,
        "clinical_summary": summary_text,
        "range_warnings": range_messages,
        "range_df": range_df,
    })

    tab1_view = st.radio(
        "Risk Hesaplayıcı Bölümü",
        T[lang]["risk_summary_radio"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if tab1_view == T[lang]["risk_summary_radio"][0]:
        metric_cols = st.columns(5)
        metric_cols[0].metric(T[lang]["risk_ratio"], f"%{risk * 100:.1f}")
        metric_cols[1].metric(T[lang]["category_label"], category)
        metric_cols[2].metric(T[lang]["threshold_label"], f"%{risk_threshold * 100:.1f}")
        metric_cols[3].metric("Sensitivity", f"{threshold_metrics['sensitivity']:.3f}")
        metric_cols[4].metric("Specificity", f"{threshold_metrics['specificity']:.3f}")
        if risk_ci:
            st.caption(f"{T[lang]['bootstrap_result']}: %{risk_ci[0] * 100:.1f} - %{risk_ci[1] * 100:.1f}")
        if category_level == "error":
            st.error(f"Sınıflandırma: {category}")
        elif category_level == "warning":
            st.warning(f"Sınıflandırma: {category}")
        else:
            st.success(f"Sınıflandırma: {category}")
        if range_messages:
            st.warning(T[lang]["range_warning"] + ": " + " ".join(range_messages))

    elif tab1_view == T[lang]["risk_summary_radio"][1]:
        st.subheader("Bootstrap " + T[lang]["bootstrap_result"])
        bootstrap_status = st.empty()
        if st.button(T[lang]["bootstrap_btn"]):
            bootstrap_status.caption(T[lang]["bootstrap_wait"])
            risk_ci = bootstrap_patient_risk_ci(patient_values, X_train, y_train)
            st.session_state["risk_ci"] = risk_ci
            bootstrap_status.empty()
        if risk_ci:
            st.caption(f"{T[lang]['bootstrap_result']}: %{risk_ci[0] * 100:.1f} - %{risk_ci[1] * 100:.1f}")
        else:
            st.info(T[lang]["bootstrap_start_info"])

    else:
        st.subheader(T[lang]["range_control"])
        if range_messages:
            st.warning(T[lang]["range_warning"] + ": " + " ".join(range_messages))
        st.dataframe(range_df, use_container_width=True)


with tab2:
    st.subheader(T[lang]["clinical_summary_title"])
    current_acog = any(current_acog_items().values())
    category, level = risk_category(st.session_state.get("risk", risk), st.session_state.get("pred", pred_class), current_acog)
    summary_text = clinical_summary(st.session_state.get("risk", risk), st.session_state.get("thresh", risk_threshold), st.session_state.get("pred", pred_class), current_acog, category)
    st.session_state["category"] = category
    st.session_state["clinical_summary"] = summary_text
    if level == "error":
        st.error(summary_text)
    elif level == "warning":
        st.warning(summary_text)
    else:
        st.success(summary_text)
    _pred = st.session_state.get('pred', pred_class)
    _risk = st.session_state.get('risk', risk)
    st.markdown(
        f"""
<div class="clinical-box">
<b>{T[lang]['risk_category_label']}:</b> {category}<br>
<b>{T[lang]['model_label']}:</b> {T[lang]['high_risk'] if _pred == 1 else T[lang]['low_risk']}<br>
<b>{T[lang]['acog_label']}:</b> {T[lang]['acog_positive'] if current_acog else T[lang]['acog_negative']}<br>
<b>{T[lang]['risk_label']}:</b> %{_risk * 100:.1f}
</div>
""",
        unsafe_allow_html=True,
    )


with tab3:
    st.subheader(T[lang]["acog_title"])
    c1, c2 = st.columns(2)
    with c1:
        st.checkbox(T[lang]["acog_bp"], key="acog_bp")
        st.checkbox(T[lang]["acog_plt"], key="acog_plt_low")
        st.checkbox(T[lang]["acog_cr"], key="acog_cr")
    with c2:
        st.checkbox(T[lang]["acog_liver"], key="acog_liver")
        st.checkbox(T[lang]["acog_pulm"], key="acog_pulmonary")
        st.checkbox(T[lang]["acog_neuro"], key="acog_neuro")
    items = current_acog_items()
    st.session_state["acog_items"] = items
    st.session_state["acog_any"] = any(items.values())
    if any(items.values()):
        st.error(T[lang]["acog_alert"])
    else:
        st.success(T[lang]["acog_ok"])


with tab4:
    st.subheader(T[lang]["shap_title"])
    shap_key = (patient_values, data_sig)
    if st.session_state.get("shap_key") != shap_key:
        st.session_state["shap_df"] = None
        st.session_state["shap_fig"] = None
        st.session_state["shap_key"] = shap_key

    if st.button(T[lang]["shap_btn"]):
        with st.spinner(T[lang]["shap_spinner"]):
            shap_df, shap_fig = shap_dataframe(
                data_sig, model, X_train_scaled,
                tuple(patient_scaled[0].tolist()), patient_values,
            )
        st.session_state["shap_df"] = shap_df
        st.session_state["shap_fig"] = shap_fig

    shap_df = st.session_state.get("shap_df")
    shap_fig = st.session_state.get("shap_fig")
    if shap_df is not None and shap_fig is not None:
        st.dataframe(shap_df[["Parametre", "Hasta Değeri", "SHAP Etkisi"]], use_container_width=True)
        st.pyplot(shap_fig)
    elif shap is None:
        st.warning(T[lang]["shap_no_pkg"])
    else:
        st.info(T[lang]["shap_info"])


with tab5:
    st.subheader(T[lang]["bulk_title"])
    st.caption(T[lang]["bulk_caption"])
    uploaded = st.file_uploader(T[lang]["bulk_upload"], type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            bulk_df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            bulk_result = predict_patients(bulk_df, scaler, model, risk_threshold)
            st.dataframe(bulk_result, use_container_width=True)
            st.download_button(
                T[lang]["bulk_csv_btn"],
                data=bulk_result.to_csv(index=False).encode("utf-8-sig"),
                file_name="toplu_preeklampsi_risk_tahminleri.csv",
                mime="text/csv",
            )
            try:
                st.download_button(
                    T[lang]["bulk_excel_btn"],
                    data=dataframe_to_excel_bytes(bulk_result),
                    file_name="toplu_preeklampsi_risk_tahminleri.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                st.info(T[lang]["bulk_excel_info"])
        except Exception as e:
            st.error(f"{T[lang]['bulk_error']}: {e}")


with tab6:
    st.subheader(T[lang]["compare_title"])
    if st.session_state.get("model_compare_key") != data_sig:
        st.session_state["model_compare_table"] = None
        st.session_state["model_compare_key"] = data_sig

    if st.button(T[lang]["compare_btn"]):
        with st.spinner(T[lang]["compare_spinner"]):
            model_table, fitted_models = train_candidate_models(
                data_sig, X_train_scaled, y_train_smote, X_test_scaled, y_test,
            )
        st.session_state["model_compare_table"] = model_table

    model_table = st.session_state.get("model_compare_table")
    if model_table is not None:
        st.dataframe(model_table, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(model_table["Model"], model_table["AUC"], color="#6b3fa0")
        ax.set_ylim(0, 1)
        ax.set_ylabel("ROC AUC")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
    else:
        st.info(T[lang]["compare_info"])
    if XGBClassifier is None:
        st.info(T[lang]["xgb_info"])


with tab7:
    st.subheader(T[lang]["calib_title"])
    frac_pos, mean_pred = calibration_curve(y_test, y_prob_test, n_bins=6, strategy="uniform")
    fig, ax = plt.subplots()
    ax.plot(mean_pred, frac_pos, marker="o", color="#6b3fa0", label=T[lang]["calib_model"])
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label=T[lang]["calib_ideal"])
    ax.set_xlabel(T[lang]["calib_xlabel"])
    ax.set_ylabel(T[lang]["calib_ylabel"])
    ax.legend()
    st.pyplot(fig)

    st.subheader(T[lang]["dca_title"])
    thresholds_dca = np.linspace(0.01, 0.99, 99)
    prevalence = y_test.mean()
    net_benefit_model = []
    net_benefit_all = []
    for th in thresholds_dca:
        pred = y_prob_test >= th
        tp_dca = np.sum((pred == 1) & (y_test.to_numpy() == 1))
        fp_dca = np.sum((pred == 1) & (y_test.to_numpy() == 0))
        n = len(y_test)
        net_benefit_model.append((tp_dca / n) - (fp_dca / n) * (th / (1 - th)))
        net_benefit_all.append(prevalence - (1 - prevalence) * (th / (1 - th)))
    fig, ax = plt.subplots()
    ax.plot(thresholds_dca, net_benefit_model, color="#6b3fa0", label=T[lang]["dca_model"])
    ax.plot(thresholds_dca, net_benefit_all, color="#d9534f", linestyle="--", label=T[lang]["dca_treat_all"])
    ax.plot(thresholds_dca, np.zeros_like(thresholds_dca), color="gray", linestyle=":", label=T[lang]["dca_treat_none"])
    ax.set_xlabel(T[lang]["dca_xlabel"])
    ax.set_ylabel(T[lang]["dca_ylabel"])
    ax.legend()
    st.pyplot(fig)


with tab8:
    st.subheader(T[lang]["data_title"])
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(T[lang]["data_total"], len(df))
    d2.metric(T[lang]["data_model"], len(data))
    d3.metric(T[lang]["data_pos"], f"%{y.mean() * 100:.1f}")
    d4.metric(T[lang]["data_missing"], int(df[FEATURES + [TARGET]].isna().sum().sum()))

    st.write(T[lang]["data_dist"])
    st.bar_chart(y.value_counts().sort_index())
    st.write(T[lang]["data_var_dist"])
    selected_feature = st.selectbox(T[lang]["data_select"], FEATURES, format_func=feature_label)
    fig, ax = plt.subplots()
    ax.hist(data[selected_feature], bins=20, color="#6b3fa0", alpha=0.8)
    ax.set_xlabel(feature_label(selected_feature))
    ax.set_ylabel(T[lang]["data_freq"])
    st.pyplot(fig)

    st.subheader(T[lang]["trend_title"])
    trend_file = st.file_uploader(T[lang]["trend_upload"], type=["csv", "xlsx", "xls"], key="trend_upload")
    if trend_file is not None:
        try:
            trend_df = pd.read_csv(trend_file) if trend_file.name.lower().endswith(".csv") else pd.read_excel(trend_file)
            trend_df = clean_numeric_dataframe(trend_df)
            st.dataframe(trend_df, use_container_width=True)
            available = [col for col in FEATURES if col in trend_df.columns]
            if available:
                fig, ax = plt.subplots(figsize=(9, 4))
                x_axis = trend_df["date"] if "date" in trend_df.columns else np.arange(1, len(trend_df) + 1)
                for col in available:
                    ax.plot(x_axis, trend_df[col], marker="o", label=feature_label(col))
                ax.legend()
                ax.set_xlabel(T[lang]["trend_time"])
                ax.set_ylabel(T[lang]["trend_value"])
                st.pyplot(fig)
            else:
                st.warning(T[lang]["trend_warning"])
        except Exception as e:
            st.error(f"{T[lang]['trend_error']}: {e}")


with tab9:
    st.subheader(T[lang]["pdf_title"])
    perf = evaluate_at_threshold(y_test, y_prob_test, st.session_state.get("thresh", risk_threshold))
    metrics = {
        "ROC AUC": auc,
        "Accuracy": perf["accuracy"],
        "Sensitivity": perf["sensitivity"],
        "Specificity": perf["specificity"],
        "PPV": perf["ppv"],
        "NPV": perf["npv"],
    }
    perf_col, pdf_col = st.columns(2)
    with perf_col:
        st.write(T[lang]["pdf_model_perf"])
        st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Değer"}), use_container_width=True)
        cm_df = pd.DataFrame(perf["cm"], index=["Gerçek Negatif", "Gerçek Pozitif"], columns=["Tahmin Negatif", "Tahmin Pozitif"])
        st.dataframe(cm_df, use_container_width=True)
        with st.expander("Classification Report"):
            st.text(classification_report(y_test, perf["pred"]))
    with pdf_col:
        include_shap = st.checkbox(T[lang]["pdf_include_shap"], value=True)
        if st.button(T[lang]["pdf_create_btn"]):
            pdf = create_pdf_report(
                st.session_state.get("patient_info", {}),
                st.session_state.get("labs", {}),
                st.session_state.get("risk", risk),
                st.session_state.get("thresh", risk_threshold),
                st.session_state.get("category", category),
                st.session_state.get("acog_any", False),
                st.session_state.get("acog_items", current_acog_items()),
                metrics,
                st.session_state.get("clinical_summary", summary_text),
                st.session_state.get("shap_fig") if include_shap else None,
                risk_ci=st.session_state.get("risk_ci"),
                range_warnings=st.session_state.get("range_warnings", []),
                model_stamp=get_model_stamp(),
            )
            st.download_button(T[lang]["pdf_download_btn"], data=pdf, file_name="preeklampsi_risk_raporu.pdf", mime="application/pdf")


with tab10:
    st.subheader(T[lang]["method_title"])
    st.write(T[lang]["method_text"])
    st.subheader(T[lang]["dict_title"])
    dictionary_df = pd.DataFrame(DATA_DICTIONARY, columns=T[lang]["dict_cols"])
    st.dataframe(dictionary_df, use_container_width=True)
    st.subheader(T[lang]["limits_title"])
    for item in T[lang]["limitations"]:
        st.markdown(f"- {item}")

    st.subheader(T[lang]["stamp_title"])
    stamp_df = pd.DataFrame(get_model_stamp().items(), columns=T[lang]["stamp_cols"])
    st.dataframe(stamp_df, use_container_width=True)

    st.subheader(T[lang]["ref_title"])
    st.dataframe(feature_range_summary(data), use_container_width=True)
