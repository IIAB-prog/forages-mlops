import pandas as pd
from src.monitoring import calculate_drift
def test_drift_report_shape():
 a=pd.DataFrame({"age_ans":[1,2,3],"profondeur_m":[10,20,30],"debit_m3h":[1.,2.,3.],"usagers_jour":[10,20,30],"jours_depuis_maintenance":[1,2,3],"pannes_12m":[0,1,0],"type_pompe":["solaire","manuelle","solaire"],"region":["Centre","Nord","Sud"]})
 b=a.copy(); r=calculate_drift(a,b); assert len(r)==8; assert "drift_flag" in r
