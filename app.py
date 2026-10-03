import json,time
from pathlib import Path
import pandas as pd, requests, streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Crypto Trading Command Center",page_icon="📈",layout="wide")
API="https://api.hyperliquid.xyz/info"
WALLET="0x8a39DB38BB3dbC79791d84147957411c2F1ef035"
DATA=Path("data"); DATA.mkdir(exist_ok=True)

def api(p,timeout=20):
    r=requests.post(API,json=p,timeout=timeout); r.raise_for_status(); return r.json()

def wallet_state():
    out={}; errs=[]
    for k,p in {
      "state":{"type":"clearinghouseState","user":WALLET},
      "spot":{"type":"spotClearinghouseState","user":WALLET},
      "orders":{"type":"openOrders","user":WALLET},
      "fills":{"type":"userFills","user":WALLET,"aggregateByTime":False},
      "portfolio":{"type":"portfolio","user":WALLET}
    }.items():
        try: out[k]=api(p)
        except Exception as e: errs.append(f"{k}: {e}")
    return out,errs

def positions(state):
    rows=[]
    for x in (state or {}).get("assetPositions",[]):
        p=x.get("position",x); s=float(p.get("szi",0) or 0)
        if abs(s)<1e-12: continue
        lev=p.get("leverage",{}) or {}
        rows.append({"Coin":p.get("coin"),"Side":"LONG" if s>0 else "SHORT",
          "Size":abs(s),"Entry":float(p.get("entryPx",0) or 0),
          "Position Value":abs(float(p.get("positionValue",0) or 0)),
          "Unrealized P&L":float(p.get("unrealizedPnl",0) or 0),
          "ROE %":float(p.get("returnOnEquity",0) or 0)*100,
          "Leverage":float(lev.get("value",0) or 0),
          "Liquidation":float(p.get("liquidationPx",0) or 0),
          "Margin Used":float(p.get("marginUsed",0) or 0)})
    return pd.DataFrame(rows)

def candles(coin,tf,limit=500):
    ms={"4h":14400000,"1d":86400000,"3d":259200000}[tf]; end=int(time.time()*1000)
    raw=api({"type":"candleSnapshot","req":{"coin":coin,"interval":tf,"startTime":end-ms*(limit+5),"endTime":end}})
    d=pd.DataFrame(raw)
    if d.empty:return d
    d["time"]=pd.to_datetime(d.t,unit="ms",utc=True)
    for c in "ohlcv": d[c]=pd.to_numeric(d[{"o":"o","h":"h","l":"l","c":"c","v":"v"}[c]],errors="coerce")
    d=d.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return d[["time","open","high","low","close","volume"]].dropna().tail(limit).reset_index(drop=True)

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def superkumo(d):
    d=d.copy()
    d["tenkan"]=(d.high.rolling(9).max()+d.low.rolling(9).min())/2
    d["kijun"]=(d.high.rolling(26).max()+d.low.rolling(26).min())/2
    d["span_a"]=(d.tenkan+d.kijun)/2
    d["span_b"]=(d.high.rolling(52).max()+d.low.rolling(52).min())/2
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    atr=rma(tr,10); hl2=(d.high+d.low)/2
    up=hl2+3*atr; lo=hl2-3*atr; stv=pd.Series(index=d.index,dtype=float); direction=pd.Series(index=d.index,dtype=float)
    for i in range(len(d)):
        if i==0: stv.iloc[i]=up.iloc[i]; direction.iloc[i]=-1
        else:
            pu,pl=up.iloc[i-1],lo.iloc[i-1]
            up.iloc[i]=min(up.iloc[i],pu) if d.close.iloc[i-1]<=pu else up.iloc[i]
            lo.iloc[i]=max(lo.iloc[i],pl) if d.close.iloc[i-1]>=pl else lo.iloc[i]
            if direction.iloc[i-1]<0:
                direction.iloc[i]=1 if d.close.iloc[i]>up.iloc[i] else -1
            else: direction.iloc[i]=-1 if d.close.iloc[i]<lo.iloc[i] else 1
            stv.iloc[i]=lo.iloc[i] if direction.iloc[i]>0 else up.iloc[i]
    plus=d.high.diff(); minus=-d.low.diff()
    p=plus.where((plus>minus)&(plus>0),0); m=minus.where((minus>plus)&(minus>0),0)
    di_p=100*rma(p,14)/rma(tr,14); di_m=100*rma(m,14)/rma(tr,14)
    dx=100*(di_p-di_m).abs()/(di_p+di_m); d["adx"]=rma(dx,14)
    d["supertrend"]=stv; d["st_dir"]=direction
    d["cloud_top"]=d[["span_a","span_b"]].max(axis=1); d["cloud_bot"]=d[["span_a","span_b"]].min(axis=1)
    d["cloud"]="ABOVE"; d.loc[d.close<d.cloud_bot,"cloud"]="BELOW"; d.loc[d.close.between(d.cloud_bot,d.cloud_top),"cloud"]="INSIDE"
    d["future_green"]=d.span_a>d.span_b
    d["signal"]="NONE"
    buy=(d.cloud=="ABOVE")&(d.st_dir>0)&(d.adx>=20)&d.future_green
    sell=(d.cloud=="BELOW")&(d.st_dir<0)&(d.adx>=20)&(~d.future_green)
    d.loc[buy,"signal"]="BUY"; d.loc[sell,"signal"]="SELL"
    return d

def tv_symbol(x):
    x=x.strip().upper()
    if ":" in x:return x
    x=x.replace("USDT.P","").replace(".P","").replace("USDT","").replace("USD","")
    return "BYBIT:"+x+"USDT.P"

def tv(sym,interval):
    cfg={"symbol":sym,"interval":interval,"theme":"dark","style":"1","locale":"en","enable_publishing":False,"hide_top_toolbar":False,"hide_legend":False,"allow_symbol_change":True,"studies":["IchimokuCloud@tv-basicstudies","Supertrend@tv-basicstudies","ADX@tv-basicstudies"],"container_id":"tvchart"}
    import json
    html=f'''<div id="tvchart" style="height:620px"></div><script src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>{json.dumps(cfg)}</script>'''
    components.html(html,height=640)

tabs=st.tabs(["📊 Overview","💼 Hyperliquid Wallet","☁️ SuperKumo","👀 Watchlist","📝 Journal"])

with tabs[0]:
    st.title("Crypto Trading Command Center")
    st.caption("Live read-only Hyperliquid tracking • SuperKumo analysis")
    try:
        w,e=wallet_state(); s=w.get("state") or {}; p=positions(s)
        ms=s.get("marginSummary",{}) or {}
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Account Value",f"$ {float(ms.get('accountValue',0) or 0):,.2f}")
        c2.metric("Perp Exposure",f"$ {float(ms.get('totalNtlPos',0) or 0):,.2f}")
        c3.metric("Margin Used",f"$ {float(ms.get('totalMarginUsed',0) or 0):,.2f}")
        c4.metric("Open Positions",len(p))
        if e: st.warning("Some wallet endpoints failed: "+" | ".join(e))
        if len(p): st.dataframe(p,use_container_width=True,hide_index=True)
        else: st.info("No open perp positions returned by Hyperliquid.")
    except Exception as ex: st.error(str(ex))

with tabs[1]:
    st.header("💼 Hyperliquid Wallet")
    st.code(WALLET)
    if st.button("🔄 Refresh live wallet",type="primary"):
        st.cache_data.clear()
    try:
        w,e=wallet_state(); s=w.get("state") or {}; ms=s.get("marginSummary",{}) or {}; p=positions(s)
        a,b,c,d=st.columns(4)
        a.metric("Equity",f"$ {float(ms.get('accountValue',0) or 0):,.2f}")
        b.metric("Position Value",f"$ {float(ms.get('totalNtlPos',0) or 0):,.2f}")
        c.metric("Margin Used",f"$ {float(ms.get('totalMarginUsed',0) or 0):,.2f}")
        d.metric("Withdrawable",f"$ {float(s.get('withdrawable',0) or 0):,.2f}")
        if len(p): st.dataframe(p,use_container_width=True,hide_index=True)
        else: st.info("No open positions returned.")
        if e: st.error("\n".join(e))
        with st.expander("Open orders"): st.dataframe(pd.DataFrame(w.get("orders") or []),use_container_width=True,hide_index=True)
        with st.expander("Recent fills"): st.dataframe(pd.DataFrame(w.get("fills") or []).head(100),use_container_width=True,hide_index=True)
    except Exception as ex: st.error(f"Wallet error: {ex}")

with tabs[2]:
    st.header("☁️ SuperKumo Analyzer")
    coin=st.text_input("Hyperliquid perp",value="ENA",placeholder="ENA / HYPE / NEAR / BTC")
    tf=st.selectbox("TradingView timeframe",["1D","4H","3D","1H"])
    if st.button("🔍 Load & Analyze",type="primary"):
        try:
            d1=superkumo(candles(coin.upper(),"1d")); h4=superkumo(candles(coin.upper(),"4h")); d3=superkumo(candles(coin.upper(),"3d"))
            if d1.empty or h4.empty: st.error("No Hyperliquid candles returned. Check the perp name.")
            else:
                s=d1.iloc[-1]; q=h4.iloc[-1]; z=d3.iloc[-1]
                daily=(s.cloud=="ABOVE" and s.st_dir>0 and s.adx>=20 and s.future_green)
                confirm=(q.st_dir>0 and q.adx>=20 and q.cloud!="BELOW")
                verdict="BUY" if daily and confirm else "WAIT" if daily else "SELL" if s.cloud=="BELOW" and s.st_dir<0 and s.adx>=20 else "HOLD"
                st.subheader(f"SuperKumo verdict: {verdict}")
                st.write("Daily direction + 4H timing + 3D context. Prefer pullbacks to Kijun/support rather than chasing.")
                x,y,zcol=st.columns(3)
                for col,title,r in [(x,"1D — Direction",s),(y,"4H — Timing",q),(zcol,"3D — Context",z)]:
                    with col:
                        st.markdown("### "+title); st.metric("Price",f"{r.close:.8g}")
                        st.write(f"Cloud: **{r.cloud}**  \\nSupertrend: **{'UP' if r.st_dir>0 else 'DOWN'}**  \\nADX: **{r.adx:.1f}**  \\nFuture cloud: **{'GREEN' if r.future_green else 'RED'}**")
                        st.caption(f"Tenkan {r.tenkan:.8g} • Kijun {r.kijun:.8g} • Supertrend {r.supertrend:.8g}")
                st.subheader("TradingView")
                tv(tv_symbol(coin),"D" if tf=="1D" else "240" if tf=="4H" else "3D" if tf=="3D" else "60")
                st.info("The TradingView widget shows built-in Ichimoku, Supertrend and ADX. The SuperKumo decision is calculated separately from Hyperliquid candles.")
                st.dataframe(d1[["time","close","tenkan","kijun","span_a","span_b","supertrend","adx","cloud","future_green","signal"]].tail(30),use_container_width=True,hide_index=True)
        except Exception as ex: st.error(f"Analysis error: {ex}")

with tabs[3]:
    st.header("👀 Watchlist")
    f=DATA/"watchlist.csv"
    df=pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["Coin","Notes","Status"])
    ed=st.data_editor(df,num_rows="dynamic",use_container_width=True,hide_index=True)
    if st.button("Save Watchlist"): ed.to_csv(f,index=False); st.success("Saved.")

with tabs[4]:
    st.header("📝 Journal")
    f=DATA/"journal.csv"
    df=pd.read_csv(f) if f.exists() else pd.DataFrame()
    ed=st.data_editor(df,num_rows="dynamic",use_container_width=True,hide_index=True)
    if st.button("Save Journal"): ed.to_csv(f,index=False); st.success("Saved.")

st.divider()
st.caption("Read-only Hyperliquid tracking. No orders are signed or submitted.")
