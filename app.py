import json,time
from pathlib import Path
import numpy as np
import pandas as pd, requests, streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

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

@st.cache_data(ttl=300)
def market_coin_map():
    """Discover HIP-3 dex markets. Failure here must never block normal analysis."""
    mapping={}
    try:
        dexs=api({"type":"perpDexs"}) or []
        for dex in dexs:
            if not dex or not isinstance(dex,dict) or not dex.get("name"):
                continue
            dex_name=str(dex["name"]).lower()
            try:
                meta=api({"type":"meta","dex":dex_name})
                for u in (meta or {}).get("universe",[]):
                    name=str(u.get("name","")).upper()
                    if name:
                        mapping[name.split(":")[-1]]=name
                        mapping[name]=name
            except Exception:
                continue
    except Exception:
        pass
    return mapping

# Known HIP-3 markets that should work even if the discovery endpoint
# temporarily returns an HTTP 500 from a hosted environment.
KNOWN_HIP3={
    "CRCL":"xyz:CRCL",
}

def resolve_coin(coin):
    c=coin.strip().upper()
    if ":" in c:
        return c
    if c in KNOWN_HIP3:
        return KNOWN_HIP3[c]
    return market_coin_map().get(c,c)

def candles(coin,tf,limit=500):
    ms={"4h":14400000,"1d":86400000,"3d":259200000}[tf]; end=int(time.time()*1000)
    api_coin=resolve_coin(coin)
    raw=api({"type":"candleSnapshot","req":{"coin":api_coin,"interval":tf,"startTime":end-ms*(limit+5),"endTime":end}})
    d=pd.DataFrame(raw)
    if d.empty:return d
    d["time"]=pd.to_datetime(d.t,unit="ms",utc=True)
    for c in "ohlcv": d[c]=pd.to_numeric(d[{"o":"o","h":"h","l":"l","c":"c","v":"v"}[c]],errors="coerce")
    d=d.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return d[["time","open","high","low","close","volume"]].dropna().tail(limit).reset_index(drop=True)

def rma(s,n):
    return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def superkumo(d):
    d=d.copy()
    # Ichimoku 9/26/52. span_a/span_b are shifted 26 candles because
    # the cloud is projected forward by 26 candles.
    d["tenkan"]=(d.high.rolling(9).max()+d.low.rolling(9).min())/2
    d["kijun"]=(d.high.rolling(26).max()+d.low.rolling(26).min())/2
    d["span_a_raw"]=(d.tenkan+d.kijun)/2
    d["span_b_raw"]=(d.high.rolling(52).max()+d.low.rolling(52).min())/2
    d["span_a"]=d.span_a_raw.shift(26)
    d["span_b"]=d.span_b_raw.shift(26)
    d["cloud_top"]=d[["span_a","span_b"]].max(axis=1)
    d["cloud_bot"]=d[["span_a","span_b"]].min(axis=1)

    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),
                  (d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    atr=rma(tr,10)
    mid=(d.high+d.low)/2
    upper=mid+3*atr
    lower=mid-3*atr
    final_u=upper.copy(); final_l=lower.copy()
    stv=pd.Series(np.nan,index=d.index); direction=pd.Series(np.nan,index=d.index)
    for i in range(len(d)):
        if i==0:
            direction.iloc[i]=1
            continue
        if pd.isna(atr.iloc[i]):
            direction.iloc[i]=direction.iloc[i-1]
            continue
        final_u.iloc[i]=upper.iloc[i] if (upper.iloc[i]<final_u.iloc[i-1] or d.close.iloc[i-1]>final_u.iloc[i-1]) else final_u.iloc[i-1]
        final_l.iloc[i]=lower.iloc[i] if (lower.iloc[i]>final_l.iloc[i-1] or d.close.iloc[i-1]<final_l.iloc[i-1]) else final_l.iloc[i-1]
        direction.iloc[i]=1 if (direction.iloc[i-1]>0 and d.close.iloc[i]>=final_l.iloc[i]) or (direction.iloc[i-1]<=0 and d.close.iloc[i]>final_u.iloc[i]) else -1
        stv.iloc[i]=final_l.iloc[i] if direction.iloc[i]>0 else final_u.iloc[i]

    plus=d.high.diff(); minus=-d.low.diff()
    p=plus.where((plus>minus)&(plus>0),0.0)
    m=minus.where((minus>plus)&(minus>0),0.0)
    dip=100*rma(p,14)/rma(tr,14); dim=100*rma(m,14)/rma(tr,14)
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    d["atr"]=atr; d["supertrend"]=stv; d["st_dir"]=direction
    d["di_plus"]=dip; d["di_minus"]=dim; d["adx"]=rma(dx,14)
    d["adx_rising"]=d.adx>d.adx.shift(1)

    d["cloud"]="INSIDE"
    d.loc[d.close>d.cloud_top,"cloud"]="ABOVE"
    d.loc[d.close<d.cloud_bot,"cloud"]="BELOW"
    d["future_green"]=d.span_a_raw>d.span_b_raw

    # 3-candle-confirmed swing levels.
    d["swing_high"]=(d.high>d.high.shift(1))&(d.high>d.high.shift(2))&(d.high>=d.high.shift(-1))&(d.high>=d.high.shift(-2))
    d["swing_low"]=(d.low<d.low.shift(1))&(d.low<d.low.shift(2))&(d.low<=d.low.shift(-1))&(d.low<=d.low.shift(-2))

    d["early_buy"]=(d.st_dir>0)&(d.st_dir.shift(1)<=0)
    d["early_sell"]=(d.st_dir<0)&(d.st_dir.shift(1)>=0)
    full_buy=(d.cloud=="ABOVE")&(d.st_dir>0)&(d.adx>=20)&d.future_green
    full_sell=(d.cloud=="BELOW")&(d.st_dir<0)&(d.adx>=20)&(~d.future_green)
    d["signal"]="NONE"
    d.loc[d.early_buy,"signal"]="EARLY BUY"
    d.loc[d.early_sell,"signal"]="EARLY SELL"
    d.loc[full_buy,"signal"]="BUY"
    d.loc[full_sell,"signal"]="SELL"
    for i in range(len(d)):
        if full_buy.iloc[i] and d.early_buy.iloc[max(0,i-8):i+1].any():
            d.iloc[i,d.columns.get_loc("signal")]="BUY MORE"

    add=(d.st_dir>0)&(d.close.shift(1)<=d.tenkan.shift(1))&(d.close>d.tenkan)&(d.low<=d.kijun*1.01)&(d.close>d.kijun)
    risky=(d.st_dir>0)&(d.adx>=20)&(d.close.shift(1)<=d.tenkan.shift(1))&(d.close>d.tenkan)&(d.low>d.kijun*1.01)
    d.loc[add&(d.signal=="NONE"),"signal"]="ADD"
    d.loc[risky&(d.signal=="NONE"),"signal"]="RISKY ADD"
    return d

def levels(d):
    # Use recent structure so an ancient low does not become the stop for a
    # current trade. On the daily chart, ~180 candles is roughly six months.
    x=d.dropna(subset=["close"]).tail(180).copy()
    price=float(x.close.iloc[-1])
    atr=float(x.atr.iloc[-1]) if pd.notna(x.atr.iloc[-1]) else price*0.03
    lows=x.loc[x.swing_low,"low"].dropna().tolist()
    highs=x.loc[x.swing_high,"high"].dropna().tolist()
    def cluster(vals):
        vals=sorted(vals); out=[]
        for v in vals:
            if not out or abs(v-out[-1])>max(atr*.75,price*.008): out.append(v)
            else: out[-1]=(out[-1]+v)/2
        return out
    supports=sorted([v for v in cluster(lows) if v<price],reverse=True)
    resistances=sorted([v for v in cluster(highs) if v>price])
    return supports[:4],resistances[:4],atr

def fmt(v):
    if v is None or pd.isna(v): return "—"
    v=float(v)
    if abs(v)>=1000:return f"{v:,.0f}"
    if abs(v)>=100:return f"{v:,.2f}"
    if abs(v)>=1:return f"{v:,.3f}"
    return f"{v:,.5f}"

def build_plan(d1,h4):
    s=d1.iloc[-1]; q=h4.iloc[-1]; price=float(s.close)
    supports,resistances,atr=levels(d1)
    daily_bull=s.cloud=="ABOVE" and s.st_dir>0 and s.adx>=20 and bool(s.future_green)
    daily_bear=s.cloud=="BELOW" and s.st_dir<0 and s.adx>=20 and not bool(s.future_green)
    h4_bull=q.st_dir>0 and q.adx>=20 and q.cloud!="BELOW"
    h4_bear=q.st_dir<0 and q.adx>=20 and q.cloud!="ABOVE"
    verdict="BUY" if daily_bull and h4_bull else "SELL" if daily_bear and h4_bear else "WAIT" if daily_bull else "HOLD"
    signal=str(s.signal)
    if signal=="NONE":
        recent=d1.tail(12); m=recent[recent.signal!="NONE"]
        signal=str(m.signal.iloc[-1]) if len(m) else "NONE"
    kijun=float(s.kijun) if pd.notna(s.kijun) else price
    pull_low=max(supports[0]*.995,kijun-atr*.35) if supports else kijun-atr*.35
    pull_high=min(kijun+atr*.20,price)
    if verdict=="BUY":
        entry=price
        candidates=[v for v in [float(s.cloud_bot) if pd.notna(s.cloud_bot) else None,supports[0] if supports else None,float(s.supertrend) if pd.notna(s.supertrend) else None] if v and v<price]
        stop=max(candidates)-atr*.15 if candidates else price-1.5*atr
        tps=resistances[:2] if len(resistances)>=2 else [price+2*atr,price+4*atr]
    elif verdict=="SELL":
        entry=price
        candidates=[v for v in [float(s.cloud_top) if pd.notna(s.cloud_top) else None,resistances[0] if resistances else None,float(s.supertrend) if pd.notna(s.supertrend) else None] if v and v>price]
        stop=min(candidates)+atr*.15 if candidates else price+1.5*atr
        tps=supports[:2] if len(supports)>=2 else [price-2*atr,price-4*atr]
    else:
        # HOLD means there is no active trade trigger. Do not manufacture an
        # entry far below spot or a misleading R:R.
        if verdict=="WAIT":
            entry=kijun
            stop=supports[0]-atr*.15 if supports else kijun-1.25*atr
        else:
            entry=np.nan
            stop=supports[0]-atr*.15 if supports else np.nan
        tps=resistances[:2] if len(resistances)>=2 else [price+2*atr,price+4*atr]
    risk=abs(entry-stop) if pd.notna(entry) and pd.notna(stop) else np.nan
    return dict(verdict=verdict,signal=signal,price=price,daily=s,h4=q,entry=entry,stop=stop,
                tp1=float(tps[0]),tp2=float(tps[1]),rr1=abs(tps[0]-entry)/risk if risk else np.nan,
                rr2=abs(tps[1]-entry)/risk if risk else np.nan,support=supports[0] if supports else None,
                resistance=resistances[0] if resistances else None,pull_low=pull_low,pull_high=pull_high)

def native_chart(d):
    x=d.tail(220)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=x.time,open=x.open,high=x.high,low=x.low,close=x.close,name="Price"))
    fig.add_trace(go.Scatter(x=x.time,y=x.tenkan,name="Tenkan 9",line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x.time,y=x.kijun,name="Kijun 26",line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=x.time,y=x.span_a,name="Cloud A",line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x.time,y=x.span_b,name="Cloud B",fill="tonexty",fillcolor="rgba(70,160,90,.16)",line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x.time,y=x.supertrend,name="Supertrend",line=dict(width=2)))
    b=x[x.signal.isin(["BUY","BUY MORE","ADD","RISKY ADD"])]
    se=x[x.signal.isin(["SELL","EARLY SELL"])]
    fig.add_trace(go.Scatter(x=b.time,y=b.low*.995,mode="markers",name="Buy signals",marker=dict(size=9,symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=se.time,y=se.high*1.005,mode="markers",name="Sell signals",marker=dict(size=9,symbol="triangle-down")))
    fig.update_layout(height=620,template="plotly_dark",margin=dict(l=10,r=10,t=30,b=10),xaxis_rangeslider_visible=False,legend=dict(orientation="h"))
    return fig
def tv_symbol(x):
    x=x.strip().upper()
    if ":" in x:return x
    return "HYPERLIQUID:"+x.replace("USDT.P","").replace(".P","").replace("USDT","")+"USDT.P"

def tv(sym,interval):
    cfg={"autosize":True,"symbol":sym,"interval":interval,"timezone":"exchange","theme":"dark","style":"1",
         "withdateranges":True,"hide_side_toolbar":False,"allow_symbol_change":True,
         "save_image":False,"studies":["IchimokuCloud@tv-basicstudies","Supertrend@tv-basicstudies","ADX@tv-basicstudies"],
         "locale":"en","support_host":"https://www.tradingview.com","calendar":False}
    payload=json.dumps(cfg)
    html=f'''<div class="tradingview-widget-container" style="height:100%;width:100%">
<div class="tradingview-widget-container__widget" style="height:680px;width:100%"></div>
<div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">TradingView</a></div>
<script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
{payload}
</script>
</div>'''
    components.html(html,height=700,scrolling=False)

tabs=st.tabs(["📊 Overview","💼 Hyperliquid Wallet","☁️ SuperKumo","👀 Watchlist","📝 Journal"])

with tabs[0]:
    st.title("Crypto Trading Command Center")
    st.caption("Live read-only Hyperliquid tracking • SuperKumo rules engine")
    try:
        w,e=wallet_state(); state=w.get("state") or {}; p=positions(state); ms=state.get("marginSummary",{}) or {}
        a,b,c,d=st.columns(4)
        a.metric("Account Value",f"$ {float(ms.get('accountValue',0) or 0):,.2f}")
        b.metric("Perp Exposure",f"$ {float(ms.get('totalNtlPos',0) or 0):,.2f}")
        c.metric("Margin Used",f"$ {float(ms.get('totalMarginUsed',0) or 0):,.2f}")
        d.metric("Open Positions",len(p))
        if e: st.warning("Some wallet endpoints failed: "+" | ".join(e))
        if len(p): st.dataframe(p,use_container_width=True,hide_index=True)
        else: st.info("No open perp positions returned by Hyperliquid.")
    except Exception as ex: st.error(str(ex))

with tabs[1]:
    st.header("💼 Hyperliquid Wallet")
    st.code(WALLET)
    if st.button("🔄 Refresh live wallet",type="primary"):
        st.cache_data.clear(); st.rerun()
    try:
        w,e=wallet_state(); state=w.get("state") or {}; ms=state.get("marginSummary",{}) or {}; p=positions(state)
        a,b,c,d=st.columns(4)
        a.metric("Equity",f"$ {float(ms.get('accountValue',0) or 0):,.2f}")
        b.metric("Position Value",f"$ {float(ms.get('totalNtlPos',0) or 0):,.2f}")
        c.metric("Margin Used",f"$ {float(ms.get('totalMarginUsed',0) or 0):,.2f}")
        d.metric("Withdrawable",f"$ {float(state.get('withdrawable',0) or 0):,.2f}")
        if len(p): st.dataframe(p,use_container_width=True,hide_index=True)
        else: st.info("No open positions returned.")
        if e: st.error("\\n".join(e))
        with st.expander("Open orders"): st.dataframe(pd.DataFrame(w.get("orders") or []),use_container_width=True,hide_index=True)
        with st.expander("Recent fills"): st.dataframe(pd.DataFrame(w.get("fills") or []).head(100),use_container_width=True,hide_index=True)
    except Exception as ex: st.error(f"Wallet error: {ex}")

with tabs[2]:
    st.header("☁️ SuperKumo Analyzer")
    st.caption("1D = direction • 4H = adds/exits • 3D = context • signals after candle close")
    a,b,c=st.columns([2,1,1])
    with a: coin=st.text_input("Hyperliquid perp",value="TAO",placeholder="TAO / ENA / HYPE / NEAR / BTC").upper().strip()
    with b: tv_tf=st.selectbox("Chart timeframe",["1D","4H","3D","1H"])
    with c:
        run=st.button("🔍 Analyze",type="primary",use_container_width=True)
    if run or st.session_state.get("sk_coin")==coin:
        st.session_state["sk_coin"]=coin
        try:
            api_coin=resolve_coin(coin)
            d1=superkumo(candles(api_coin,"1d")); h4=superkumo(candles(api_coin,"4h")); d3=superkumo(candles(api_coin,"3d"))
            if d1.empty or h4.empty or d3.empty: st.error("No Hyperliquid candles returned. Check the perp name.")
            else:
                st.caption("Hyperliquid API market: " + api_coin)
                p=build_plan(d1,h4); s,q,z=p["daily"],p["h4"],d3.iloc[-1]
                st.divider()
                x1,x2,x3,x4=st.columns(4)
                x1.metric("VERDICT",p["verdict"]); x2.metric("SIGNAL",p["signal"]); x3.metric("PRICE",fmt(p["price"])); x4.metric("DAILY ADX",f"{s.adx:.1f}")
                st.markdown("### Decision")
                st.write(f"**Daily:** {s.cloud} cloud • Supertrend {'UP' if s.st_dir>0 else 'DOWN'} • ADX {s.adx:.1f} • Future cloud {'GREEN' if s.future_green else 'RED'}")
                st.write(f"**4H:** {q.cloud} cloud • Supertrend {'UP' if q.st_dir>0 else 'DOWN'} • ADX {q.adx:.1f}")
                st.write(f"**3D:** {z.cloud} cloud • Supertrend {'UP' if z.st_dir>0 else 'DOWN'} • ADX {z.adx:.1f}")
                a,b,c,d=st.columns(4)
                a.metric("Entry / Trigger",fmt(p["entry"])); b.metric("Stop",fmt(p["stop"])); c.metric("TP1",fmt(p["tp1"])); d.metric("TP2",fmt(p["tp2"]))
                a,b,c,d=st.columns(4)
                a.metric("R:R → TP1",f"{p['rr1']:.2f}"); b.metric("R:R → TP2",f"{p['rr2']:.2f}"); c.metric("Support",fmt(p["support"])); d.metric("Resistance",fmt(p["resistance"]))
                st.info(f"Preferred pullback zone: {fmt(p['pull_low'])} – {fmt(p['pull_high'])}. Avoid chasing extended candles.")
                reasons=[
                    "price above daily cloud" if s.cloud=="ABOVE" else "price below daily cloud" if s.cloud=="BELOW" else "price inside daily cloud",
                    "daily Supertrend UP" if s.st_dir>0 else "daily Supertrend DOWN",
                    f"ADX {'passing' if s.adx>=20 else 'below'} 20",
                    "future cloud green" if s.future_green else "future cloud red",
                    "4H confirms" if (q.st_dir>0 and q.adx>=20) else "4H not fully confirming"
                ]
                st.markdown("**Why:** "+" • ".join(reasons)+".")
                st.subheader("SuperKumo chart — Hyperliquid data")
                st.plotly_chart(native_chart(d1),use_container_width=True)
                st.subheader("TradingView")
                interval={"1D":"D","4H":"240","3D":"3D","1H":"60"}[tv_tf]
                tv(tv_symbol(coin),interval)
                st.caption("The TradingView chart uses the Hyperliquid feed. The SuperKumo verdict and levels above are independently calculated from Hyperliquid candles.")
                with st.expander("Indicator values"):
                    st.dataframe(d1[["time","close","tenkan","kijun","span_a","span_b","supertrend","adx","cloud","future_green","signal"]].tail(40),use_container_width=True,hide_index=True)
                with st.expander("SuperKumo rules"):
                    st.markdown("""
**EARLY BUY:** Supertrend flip up.  
**BUY:** above cloud + Supertrend up + ADX >= 20 + future cloud green.  
**BUY MORE:** BUY within 8 candles of EARLY BUY.  
**ADD:** uptrend + Kijun pullback + close back above Tenkan.  
**RISKY ADD:** shallow Tenkan reclaim with ADX >= 20.  
**SELL:** inverse conditions.  
**S/R:** 3-candle-confirmed swing highs/lows.
""")
        except requests.HTTPError as ex:
            st.error("SuperKumo API error: " + str(ex) + ". The app now resolves HIP-3 tickers to their DEX-qualified market name automatically.")
        except Exception as ex: st.error(f"SuperKumo error: {ex}")



with tabs[3]:
    st.header("👀 Watchlist")
    f=DATA/"watchlist.csv"
    df=pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["Coin","Notes","Status"])
    ed=st.data_editor(df,num_rows="dynamic",use_container_width=True,hide_index=True)
    if st.button("Save Watchlist"):
        ed.to_csv(f,index=False); st.success("Saved.")

with tabs[4]:
    st.header("📝 Journal")
    f=DATA/"journal.csv"
    df=pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["Date","Coin","Horizon","Direction","Entry","Exit","P&L","Notes"])
    ed=st.data_editor(df,num_rows="dynamic",use_container_width=True,hide_index=True)
    if st.button("Save Journal"):
        ed.to_csv(f,index=False); st.success("Saved.")

st.divider()
st.caption("Read-only Hyperliquid tracking. No orders are signed or submitted.")
