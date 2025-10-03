
import os, numpy as np, pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data", "synthetic_mmm_data.csv")
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

def adstock(x, decay):
    y = np.zeros_like(x, dtype=float); carry=0.0
    for i, v in enumerate(x):
        carry = v + decay*carry; y[i]=carry
    return y
def hill(x, alpha=1.0, lam=1.0):
    return alpha * x / (lam + x + 1e-9)

def build_features(df):
    params = {
        "tv_spend":      (0.7, 0.0006, 120000),
        "search_spend":  (0.4, 0.0010, 20000),
        "social_spend":  (0.5, 0.0009, 18000),
        "display_spend": (0.3, 0.0007, 15000),
        "email_spend":   (0.2, 0.0012, 8000),
    }
    cols = {}
    for ch,(decay,alpha,lam) in params.items():
        ad = adstock(df[ch].values, decay)
        cols[f"{ch}_adstock"]=ad
        cols[f"{ch}_sat"]=hill(ad, alpha, lam)
    X = pd.DataFrame(cols)
    for c in ["promo","price_index","competitor_index","seasonality","holiday_boost"]:
        X[c]=df[c].values
    y = df["sales"].values
    return X,y

def fit_models(X,y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = GridSearchCV(Pipeline([("scaler",StandardScaler()),("model",Ridge())]),
                         {"model__alpha":[0.1,1,3,10,30]}, cv=cv, scoring="neg_mean_squared_error").fit(X,y)
    lasso = GridSearchCV(Pipeline([("scaler",StandardScaler()),("model",Lasso(max_iter=10000))]),
                         {"model__alpha":[0.0005,0.001,0.005,0.01]}, cv=cv, scoring="neg_mean_squared_error").fit(X,y)
    rf = RandomForestRegressor(n_estimators=600, random_state=42).fit(X,y)
    return {"ridge":ridge,"lasso":lasso,"random_forest":rf}

def evaluate(models,X,y):
    rows=[]
    for name,m in models.items():
        est = m.best_estimator_ if hasattr(m,"best_estimator_") else m
        yhat = est.predict(X)
        rmse = np.sqrt(np.mean((y-yhat)**2))
        r2 = 1 - np.sum((y-yhat)**2)/np.sum((y - y.mean())**2)
        rows.append({"model":name,"rmse":rmse,"r2":r2})
    df = pd.DataFrame(rows).sort_values("rmse")
    df.to_csv(os.path.join(OUT,"model_summary.csv"), index=False)
    return df

def channel_contributions(model,X,df_raw):
    baseline = model.predict(X*0 + X.mean())
    contribs={}
    channel_cols=[c for c in X.columns if c.endswith("_sat")]
    for col in channel_cols:
        Xm=X.copy(); Xm[col]=X[col]
        yhat=model.predict(Xm)
        contribs[col]=(yhat-baseline).mean()
    cc = pd.DataFrame({"channel":list(contribs.keys()),"avg_weekly_contribution":list(contribs.values())})
    spend_map = {ch: df_raw[ch.replace("_sat","").replace("_adstock","")].mean() for ch in channel_cols}
    cc["avg_weekly_spend"]=cc["channel"].map(spend_map)
    cc["roi_proxy"]=cc["avg_weekly_contribution"]/(cc["avg_weekly_spend"]+1e-9)
    cc.to_csv(os.path.join(OUT,"channel_contributions.csv"), index=False)
    return cc

def budget_optimizer(model, df_raw, X, total_delta=50000.0):
    channel_cols=[c for c in X.columns if c.endswith("_sat")]
    # crude gradient-based linearized optimizer around mean
    from sklearn.linear_model import Ridge
    lin = Ridge(alpha=1.0).fit(X, df_raw["sales"].values)
    coef_map = {f:c for f,c in zip(X.columns, lin.coef_)}
    grads = np.array([coef_map[c] for c in channel_cols])
    caps = np.full_like(grads, total_delta/2)
    def objective(d): return -np.sum(grads*d)
    cons = {"type":"eq","fun":lambda d: np.sum(d)-total_delta}
    bnds=[(0,caps[i]) for i in range(len(grads))]
    x0 = np.full_like(grads, total_delta/len(grads))
    res = minimize(objective, x0, bounds=bnds, constraints=cons, method="SLSQP")
    rec = pd.DataFrame({"channel":channel_cols, "recommended_incremental_spend":res.x})
    rec.to_csv(os.path.join(OUT,"budget_recommendations.csv"), index=False)
    return rec

def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    X,y = build_features(df)
    models = fit_models(X,y)
    summary = evaluate(models,X,y)
    best_name = summary.iloc[0]["model"]
    best = models[best_name].best_estimator_ if hasattr(models[best_name],"best_estimator_") else models[best_name]
    cc = channel_contributions(best,X,df)
    rec = budget_optimizer(best,df,X, total_delta=50000.0)
    print(summary.to_string(index=False))
    print("\nTop contributions:\n", cc.sort_values("avg_weekly_contribution", ascending=False).head(5).to_string(index=False))
    print("\nBudget recommendations (sum=$50k):\n", rec.to_string(index=False))

if __name__ == "__main__":
    main()
