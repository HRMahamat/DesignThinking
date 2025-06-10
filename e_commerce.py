import os
import time
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler as SC
from scipy.sparse import coo_matrix
import implicit

# ------------------------------------------------------------------
# 0️⃣ – CONFIG & CSS
# ------------------------------------------------------------------
st.set_page_config(
    page_title="📊 E-Commerce Cameroon IA 📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.sidebar .sidebar-content { background: #001f3f; color: #fff; }
.stButton>button, .stDownloadButton>button { border-radius:8px; font-weight:bold; }
.stButton>button { background: #0074D9; color:#fff; }
.stDownloadButton>button { background: #2ECC40; color:#fff; }
h1, h2, h3 { color: #001f3f; }
.block-container { padding:1rem 2rem; }
.css-1d391kg { background: #fff; border-radius:8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1); padding:1rem; }
</style>
""", unsafe_allow_html=True)

ass = ["Sélectionnez la tranche d'âge", "-18", "18-24", "25-34", "35-44", "45+"]
ms  = ["Sélectionnez le mois", "Aucun",
       "Janvier","Février","Mars","Avril","Mai","Juin",
       "Juillet","Aout","Septembre","Octobre","Novembre","Décembre"]

# ------------------------------------------------------------------
# 1️⃣ – CHARGEMENT & NETTOYAGE (pandas only)
# ------------------------------------------------------------------
@st.cache_data
def load_and_clean():
    df = pd.read_csv("Avis clients.csv", sep=";")
    df.drop(columns=["Horodateur"], inplace=True)

    # safe rename colonnes
    df.columns = (
        df.columns
        .str.replace(r'[^0-9A-Za-z]', '_', regex=True)
        .str.strip('_')
    )

    # ages_bucket
    age_col = [c for c in df.columns if "Quel__ge" in c][0]
    df["ages_bucket"] = (
        df[age_col]
        .replace(r".*18-24.*", "18-24", regex=True)
        .replace(r".*25-34.*", "25-34", regex=True)
        .replace(r".*35-44.*", "35-44", regex=True)
        .replace(r".*45.*",   "45+",   regex=True)
        .fillna("-18")
    )

    # month_pref
    raw = [c for c in df.columns if "privil_giez" in c][0]
    df["month_pref"] = (
        df[raw]
        .replace(r".*(Janvier).*",  "Janvier",  regex=True)
        .replace(r".*(F[ée]vrier).*", "Février", regex=True)
        .replace(r".*(Mars).*",      "Mars",    regex=True)
        .replace(r".*(Avril).*",     "Avril",   regex=True)
        .replace(r".*(Mai).*",       "Mai",     regex=True)
        .replace(r".*(Juin).*",      "Juin",    regex=True)
        .replace(r".*(Juillet).*",   "Juillet", regex=True)
        .replace(r".*(Ao[ûu]t).*",   "Aout",    regex=True)
        .replace(r".*(Septembre).*", "Septembre", regex=True)
        .replace(r".*(Octobre).*",   "Octobre",   regex=True)
        .replace(r".*(Novembre).*",  "Novembre",  regex=True)
        .replace(r".*(D[ée]cembre).*","Décembre", regex=True)
        .fillna("Autre")
    )

    # Abandon flag
    ab_col = [c for c in df.columns if "abandonn" in c.lower()][0]
    df["Abandon_flag"] = df[ab_col].apply(
        lambda x: 0 if str(x).strip().lower().startswith("non") else 1
    )

    return df

pdf_raw = load_and_clean()

# ------------------------------------------------------------------
# 2️⃣ – SIDEBAR & FILTRES
# ------------------------------------------------------------------
st.sidebar.title("📊 CommerceGenius")
page = st.sidebar.radio("", [
    "Accueil","Analytics Live","Segmentation",
    "Recommandations","Alertes","Visualisations",
    "Export CSV","Commentaires"
])
st.sidebar.markdown("---")

age_col     = "ages_bucket"
city_col    = [c for c in pdf_raw.columns if "ville_habitez" in c.lower()][0]
month_col   = "month_pref"
achat_col   = [c for c in pdf_raw.columns if "fr_quence" in c.lower()][0]
mode_col    = [c for c in pdf_raw.columns if "paiement" in c.lower()][0]
product_col = [c for c in pdf_raw.columns if "achetez_habituellement" in c.lower()][0]

sel_raw = pdf_raw.copy()
a = st.sidebar.selectbox("Tranche d'âge", ass)
if a!=ass[0]: sel_raw = sel_raw[ sel_raw[age_col]==a ]
v = st.sidebar.selectbox("Ville", ["Sélectionnez la ville", "Bafoussam", "Bamenda", "Bertoua", "Douala", "Garoua", "Kumba", "Limbé", "Maroua", "Ngoundéré", "Nkongsamba", "Yaoundé"])
if v!="Sélectionnez la ville": sel_raw = sel_raw[ sel_raw[city_col]==v ]
m = st.sidebar.selectbox("Période", ms)
if m not in (ms[0],ms[1]): sel_raw = sel_raw[ sel_raw[month_col]==m ]

# ------------------------------------------------------------------
# 3️⃣ – ACCUEIL
# ------------------------------------------------------------------
if page=="Accueil":
    st.markdown("\n\n## 🎯 CommerceGenius – Comportement Client")
    st.markdown("Tableau de bord E-Commerce Cameroun : en temps réel, segmentation, recommandations.")
    img = Image.open("image.jpg")
    st.image(img.resize((1000, int((float(img.size[1]) * float((700 / float(img.size[0])))))), Image.FILTERED), use_container_width=False)

# ------------------------------------------------------------------
# 4️⃣ – ANALYTICS LIVE
# ------------------------------------------------------------------
elif page=="Analytics Live":
    st.markdown("\n\n## 📈 Analytics en Temps Réel")
    counts = pdf_raw[age_col].value_counts().reindex(ass[1:]).fillna(0).reset_index()
    counts.columns = [age_col, "count"]
    fig = px.bar(counts, x=age_col, y="count",
                 title="Répartition par tranche d'âge",
                 labels={age_col:"Âge","count":"Clients"})
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 5️⃣ – SEGMENTATION DYNAMIQUE
# ------------------------------------------------------------------
elif page=="Segmentation":
    st.markdown("\n\n## 🔍 Segmentation Dynamique (KMeans)")
    df = sel_raw[[age_col, month_col]].dropna().copy()
    df["age_idx"],   _ = pd.factorize(df[age_col])
    df["month_idx"],_ = pd.factorize(df[month_col])
    X = SC().fit_transform(df[["age_idx","month_idx"]])

    sils=[]
    for k in range(2,21):
        km=KMeans(n_clusters=k,random_state=42).fit(X)
        sils.append((k, silhouette_score(X, km.labels_)))
    sil_df=pd.DataFrame(sils,columns=["k","silhouette"])
    st.plotly_chart(px.line(sil_df,x="k",y="silhouette",markers=True),use_container_width=True)

    best_k=int(sil_df.loc[sil_df.silhouette.idxmax(),"k"])
    st.markdown(f"**👉 k optimal : {best_k}**")
    km=KMeans(n_clusters=best_k,random_state=42).fit(X)
    df["cluster"]=km.labels_
    st.plotly_chart(px.scatter(df, x="age_idx", y="month_idx", color="cluster"), use_container_width=True)

# ------------------------------------------------------------------
# 6️⃣ – RECOMMANDATIONS PERSO
# ------------------------------------------------------------------
elif page=="Recommandations":
    st.markdown("\n\n## 🤖 Recommandations Personnalisées")
    df_seg = sel_raw[["Nom_d_utilisateur", product_col]].dropna().copy()
    if sel_raw.empty:
        st.warning("⚠️ Aucun utilisateur ne correspond à vos filtres.")
    else:
        df_seg["rating"] = 1
        df_seg["user_id"], users = pd.factorize(df_seg["Nom_d_utilisateur"])
        df_seg["item_id"], items = pd.factorize(df_seg[product_col])

        M = coo_matrix(
            (df_seg["rating"], (df_seg["user_id"], df_seg["item_id"])),
            shape=(len(users), len(items))
        )

        model_seg = implicit.als.AlternatingLeastSquares(
            factors=20, regularization=0.1, iterations=20, random_state=42
        )
        model_seg.fit(M.T)

        valid_uids = [u for u in df_seg["user_id"].unique()
                      if u < model_seg.user_factors.shape[0]]
        if not valid_uids:
            st.warning("⚠️ Pas assez de données pour profiler le segment.")
        else:
            segment_vec = model_seg.user_factors[valid_uids].mean(axis=0)
            scores = model_seg.item_factors.dot(segment_vec)

            top_n = 5
            top_idx = np.argsort(scores)[::-1][:top_n]
            recs = [(items[i], float(scores[i])) for i in top_idx]

            df_recs = pd.DataFrame(recs, columns=["Produit", "Score"])
            st.markdown(f"### 🎁 Top {top_n} recommandations pour votre segment avec : Rapidité de livraison (sans frais) et Respect de la transparence des produits")
            st.table(df_recs.style.format({"Score": "{:.2f}"}))

# ------------------------------------------------------------------
# 7️⃣ – ALERTES AUTOMATIQUES
# ------------------------------------------------------------------
elif page=="Alertes":
    st.markdown("\n\n## 🚨 Alertes Comportement")
    df_a = sel_raw[[age_col,month_col,achat_col,mode_col,"Abandon_flag"]].dropna().copy()
    for c in [age_col,month_col,achat_col,mode_col]:
        df_a[f"{c}_idx"],_ = pd.factorize(df_a[c])
    feats = df_a[[f"{c}_idx" for c in [age_col,month_col,achat_col,mode_col]]+["Abandon_flag"]]
    iso = IsolationForest(contamination=0.05,random_state=42)
    df_a["anomaly"]=iso.fit_predict(feats)
    out=df_a[df_a.anomaly==-1]
    rate=len(out)/len(df_a)*100
    st.metric("Taux d’anomalies",f"{rate:.1f}%")
    if not out.empty:
        st.dataframe(out.head(10))
    else:
        st.success("✅ Aucune anomalie détectée.")

# ------------------------------------------------------------------
# 8️⃣ – VISUALISATIONS INTERACTIVES
# ------------------------------------------------------------------
elif page=="Visualisations":
    st.markdown("\n\n## 📊 Visualisations")
    if sel_raw.empty:
        st.warning("⚠️ Pas de données.")
    else:
        st.plotly_chart(px.histogram(sel_raw, x=achat_col, category_orders={achat_col:["Jamais","Une fois par mois","Plusieurs fois par mois",  "Hebdomadairement", "Quotidiennement"]}),use_container_width=True)
        df_pay = sel_raw[mode_col].value_counts().reset_index()
        df_pay.columns = ["Mode", "Nombre"]
        fig2 = px.bar(df_pay, x="Mode", y="Nombre", title="Top modes de paiement", labels={"Mode":"Mode de paiement","Nombre":"Nombre de réponses"})
        fig2.update_layout(xaxis_tickangle=-45, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# 9️⃣ – EXPORT CSV
# ------------------------------------------------------------------
elif page=="Export CSV":
    st.markdown("\n\n## 📥 Export des données filtrées")
    st.download_button("⬇️ Télécharger CSV", sel_raw.to_csv(index=False), "export.csv","text/csv")

# ------------------------------------------------------------------
# 🔟 – COMMENTAIRES
# ------------------------------------------------------------------
else:
    st.markdown("\n\n## 💬 Nos Commentaires")
    txt=st.text_area("Commentaires…")
    if st.button("Ajouter"):
        with open("comments.txt","a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} — {txt}\n")
        st.success("👍")
    if os.path.exists("comments.txt"):
        st.text(open("comments.txt", "r", encoding="utf-8").read())
