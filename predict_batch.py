import pandas as pd
import joblib
from config import DATA_DIR, MODEL_DIR, FEATURES, TARGET, REPORT_DIR
model=joblib.load(MODEL_DIR/"forage_risk_model.joblib")
df=pd.read_csv(DATA_DIR/"forages_nouveau_mois.csv")
X=df[FEATURES] # la cible panne_90j est exclue des entrées
df["risque_panne_90j"]=model.predict_proba(X)[:,1]
df["classe_risque"]=pd.cut(df.risque_panne_90j,bins=[-0.001,0.4,0.7,1.001],labels=["faible","modere","eleve"]).astype(str)
REPORT_DIR.mkdir(exist_ok=True); df.sort_values("risque_panne_90j",ascending=False).to_csv(REPORT_DIR/"forages_scores.csv",index=False)
print(f"Scores calculés pour {len(df)} forages -> reports/forages_scores.csv")
